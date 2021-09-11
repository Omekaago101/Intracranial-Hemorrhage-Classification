import torch
import torch.nn as nn
from model import WaveletLeTransform
from config.config import get_cfg
from utils import *
from solver import make_lr_scheduler, make_optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from apex import amp
from sklearn.metrics import roc_auc_score, log_loss
import argparse
import os
from datasets.rsna_dataset import RSNAHemorrhageDS3d
from class_weighted_bce_loss import WeightedBCEWithLogitsLoss
import numpy as np
import pandas as pd
from datasets.custom_dataset import IntracranialDataset
import glob
from einops import rearrange
import gc
from torch.utils.tensorboard import SummaryWriter
import sys
from torchvision.utils import make_grid
from torchsampler.weighted_sampler import ImbalancedDatasetSampler as imb
import pandas as pd
from sklearn.utils import shuffle
import warnings
# from torchsampler.imbalanced_sampler import ImbalancedDatasetSampler as imb

warnings.filterwarnings('ignore')
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
print(f'{TORCH_MAJOR}')
print(f'{TORCH_MINOR}')
writer = SummaryWriter("runs/wavelet")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
                        help="config yaml path")
    parser.add_argument("--load", type=str, default="./weights/RSNA_Wavelet_Transformer.pth",
                        help="path to model weight")
    parser.add_argument("--fold", type=int, default=0,
                        help="fold for validation")
    parser.add_argument("-ft", "--finetune", action="store_true",
                        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
                        help="model running mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
                        help="enable evaluation mode for validation", )
    parser.add_argument("--test", action="store_true",
                        help="enable evaluation mode for testset")
    parser.add_argument("--tta", action="store_true",
                        help="enable tta infer")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="enable debug mode for test")
    parser.add_argument('-y', '--autocrop', action="store",
                        dest="autocrop", help="Autocrop", default="T")
    parser.add_argument('-s', '--seed', action="store",
                        dest="seed", help="model seed", default="1234")
    parser.add_argument('-p', '--nbags', action="store",
                        dest="nbags", help="Number of bags for averaging", default="0")
    parser.add_argument('-e', '--epochs', action="store",
                        dest="epochs", help="epochs", default="5")
    parser.add_argument('-j', '--start', action="store",
                        dest="start", help="Start epochs", default="0")
    parser.add_argument('-w', '--workpath', action="store",
                        dest="workpath", help="Working path", default="densenetv1/weights")
    parser.add_argument('-f', '--weightsname', action="store",
                        dest="weightsname", help="Weights file name", default="pytorch_model.bin")
    parser.add_argument('-g', '--logmsg', action="store",
                        dest="logmsg", help="root directory", default="Recursion-pytorch")
    parser.add_argument('-c', '--size', action="store",
                        dest="size", help="model size", default="512")
    parser.add_argument('-a', '--infer', action="store",
                        dest="infer", help="root directory", default="TRN")
    parser.add_argument('-z', '--wtsize', action="store",
                        dest="wtsize", help="model size", default="999")
    parser.add_argument('-hf', '--hflip', action="store",
                        dest="hflip", help="Augmentation - Embedding horizontal flip", default="F")
    parser.add_argument('-tp', '--transpose', action="store",
                        dest="transpose", help="Augmentation - Embedding transpose", default="F")
    parser.add_argument('-xg', '--stage2', action="store",
                        dest="stage2", help="Stage2 embeddings only", default="F")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args

def build_model(cfg):
    model = WaveletLeTransform(cfg.MODEL.WL_CHNS, cfg.MODEL.CONV_CHNS, cfg.MODEL.LEVELS)
    return model


def dataloader(cfg, df, autocrop, hflip, transpose, class_props, intra_class_props, mode='train'):
    DataSet = IntracranialDataset
    data_loader = ""
    if mode == 'train':
        train_img_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN)
        train_ds = DataSet(cfg, df, train_img_path, labels=True, AUTOCROP=autocrop, HFLIP=hflip, TRANSPOSE=transpose,
                           mode="train")
        if cfg.DEBUG:
            train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 50))
        data_loader = DataLoader(train_ds, 4, sampler=imb(df, class_props, intra_class_props),
                                 pin_memory=True, shuffle=False,
                                 drop_last=False, num_workers=1)

    elif mode == 'valid':
        valid_img_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN)
        valid_ds = DataSet(cfg, df, valid_img_path, labels=True, AUTOCROP=autocrop, HFLIP=hflip, TRANSPOSE=transpose,
                           mode="valid")
        if cfg.DEBUG:
            valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 20))
        data_loader = DataLoader(valid_ds, 4, sampler=imb(df, class_props, intra_class_props),
                                     pin_memory=True, shuffle=False,
                                     drop_last=False, num_workers=2)
    # test_img_path = os.path.join(cfg.DIRS.DATA,cfg.DIRS.TEST)

    # test_ds = DataSet(cfg,test,test_img_path,labels=True,AUTOCROP=autocrop,HFLIP=hflip,TRANSPOSE=transpose, mode="test")

    return data_loader


def train_loop(_print, cfg, train_criterion, valid_criterion, class_props, intra_class_props, train, AUTOCROP, HFLIP,
               TRANSPOSE, start_epoch, best_metric):
    # Create model
    model = build_model(cfg)
    optimizer = make_optimizer(cfg, model)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()

    '''
    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                          opt_level=cfg.SYSTEM.OPT_L,
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))
    '''

    folds = [0, 1, 2, 3, 4]
    for i in range(len(folds)):
        valdf = train[train['fold'] == i].reset_index(drop=True)
        trndf = train[train['fold'] != i].reset_index(drop=True)

        # shuffle the train df
        trndf = shuffle(trndf, random_state=48)

        # split trndf into 6 parts
        train_csvs = np.array_split(trndf, 5)

        # reset the index of each dataframe
        train_csvs = [df.reset_index() for df in train_csvs]
        valid_loader = dataloader(cfg, valdf, AUTOCROP, HFLIP, TRANSPOSE, class_props, intra_class_props, mode="valid")

        for k, train in enumerate(train_csvs):
            train_loader = dataloader(cfg, train, AUTOCROP, HFLIP, TRANSPOSE, class_props, intra_class_props,
                                      mode="train")

            scheduler = make_lr_scheduler(cfg, optimizer, train_loader)

            # Load checkpoint
            try:
                if args.load != "":
                    if os.path.isfile(args.load):
                        _print(f"=> loading checkpoint {args.load}")
                        ckpt = torch.load(args.load, "cpu")
                        model.load_state_dict(ckpt.pop('state_dict'))
                        if not args.finetune:
                            _print("resuming optimizer ...")
                            optimizer.load_state_dict(ckpt.pop('optimizer'))
                            start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
                        _print(
                            f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
                    else:
                        _print(f"=> no checkpoint found at '{args.load}'")
            except FileNotFoundError:
                pass

            for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
                _print(f"\nEpoch {epoch + 1} for train_{k + 1}")
                mixed_target = ""
                lamb = ""

                losses = AverageMeter()
                model.train()
                tbar = tqdm(train_loader)

                torch.cuda.empty_cache()

                for x, batch in enumerate(tbar):
                    image = batch["image"].cuda()
                    target = batch["labels"].cuda()
                    image = rearrange(image, 'b w h c->b c w h')

                    # calculate loss
                    if cfg.DATA.CUTMIX:
                        image, target, mixed_target, lamb = cutmix_data(image, target, cfg.DATA.CM_ALPHA)
                    elif cfg.DATA.MIXUP:
                        image, target, mixed_target, lamb = mixup_data(image, target, cfg.DATA.CM_ALPHA)
                    output = model(image)
                    target, mixed_target = target.type(torch.float32), mixed_target.type(torch.float32)
                    loss = mixup_criterion(train_criterion, output, target, mixed_target, lamb)

                    del target, mixed_target, lamb, output
                    gc.collect()
                    torch.cuda.empty_cache()
                    # gradient accumulation
                    loss = loss / cfg.OPT.GD_STEPS
                    loss.backward()
                    '''
                    if cfg.SYSTEM.FP16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    '''
                    if (x + 1) % cfg.OPT.GD_STEPS == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    # record loss
                    losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
                    tbar.set_description(
                        "Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))
                    writer.add_scalar("Train/Loss", loss.item(), epoch)
                _print("Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

                if epoch%2 == 0:
                    if valid_loader is not None:
                        loss = valid_model(_print, cfg, model, valid_loader, valid_criterion)
                        is_best = loss < best_metric
                        best_metric = min(loss, best_metric)
                        writer.add_scalar("valid/Loss",loss.item(),epoch)
                    save_checkpoint({
                        "epoch": epoch + 1,
                        "arch": cfg.EXP,
                        "state_dict": model.state_dict(),
                        "best_metric": best_metric,
                        "optimizer": optimizer.state_dict(),
                    }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")


def valid_model(_print, cfg, model, valid_loader, valid_criterion):
    folds = [0, 1, 2, 3, 4]
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, batch in enumerate(tbar):
            image = batch['image'].cuda()
            target = batch['labels'].cuda()
            image = rearrange(image, 'b w h c->b c w h')
            output = model(image)
            preds.append(output.cpu())
            targets.append(target.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    targets = targets.type(torch.float32)
    # record loss
    loss_tensor = valid_criterion(preds, targets)
    val_loss = loss_tensor.sum() / valid_criterion.class_weights.sum()
    any_loss = loss_tensor[0]
    intraparenchymal_loss = loss_tensor[1]
    intraventricular_loss = loss_tensor[2]
    subarachnoid_loss = loss_tensor[3]
    subdural_loss = loss_tensor[4]
    epidural_loss = loss_tensor[5]
    _print(
        "Val. loss: %.5f - any: %.3f - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
            val_loss, any_loss,
            intraparenchymal_loss, intraventricular_loss,
            subarachnoid_loss, subdural_loss, epidural_loss))
    # record AUC
    auc = roc_auc_score(targets[:, 1:].numpy(), preds[:, 1:].numpy(), average=None)
    _print(
        "Val. AUC - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
            auc[0], auc[1], auc[2], auc[3], auc[4]))
    return val_loss


def test_model(_print, cfg, model, test_loader):
    # switch to evaluate mode
    model.eval()

    ids = []
    probs = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            id_code = list(*zip(*id_code))
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            output = model(image, seq_len)
            output = torch.sigmoid(output)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = ["image", "any",
                      "intraparenchymal", "intraventricular",
                      "subarachnoid", "subdural", "epidural"]
    return submit


def create_submission(pred_df, sub_fpath):
    imgid = pred_df["image"].values
    output = pred_df.loc[:, pred_df.columns[1:]].values
    data = [[iid] + [sub_o for sub_o in o] for iid, o in zip(imgid, output)]
    table_data = []
    for subdata in data:
        table_data.append([subdata[0] + '_any', subdata[1]])
        table_data.append([subdata[0] + '_intraparenchymal', subdata[2]])
        table_data.append([subdata[0] + '_intraventricular', subdata[3]])
        table_data.append([subdata[0] + '_subarachnoid', subdata[4]])
        table_data.append([subdata[0] + '_subdural', subdata[5]])
        table_data.append([subdata[0] + '_epidural', subdata[6]])
    df = pd.DataFrame(data=table_data, columns=['ID', 'Label'])
    df.to_csv(f'{sub_fpath}.csv', index=False)


def main(args, cfg):
    torch.cuda.empty_cache()
    # Set logger
    logging = setup_logger(args.mode, cfg.DIRS.LOGS, 0, filename=f"{cfg.EXP}.txt")
    AUTOCROP = args.autocrop
    HFLIP = 'T' if args.hflip == 'T' else ''
    TRANSPOSE = 'P' if args.transpose == 'T' else ''
    # Declare variables
    start_epoch = 0
    best_metric = 10.

    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.LOSS.WEIGHTS))
    valid_criterion = WeightedBCEWithLogitsLoss(class_weights=torch.tensor(cfg.LOSS.WEIGHTS), reduction='none')

    train = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN_CSV))
    test = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TEST_CSV))

    train_png = glob.glob(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN, '*.jpg'))
    train_png = [os.path.basename(png)[:-4] for png in train_png]

    train_imgs = set(train.Image.tolist())
    t_png = [p for p in train_png if p in train_imgs]
    t_png = np.array(t_png)
    train = train.set_index('Image').loc[t_png].reset_index()

    del train_png, t_png
    gc.collect()

    class_props = [0.1, 0.27, 0.15, 0.20, 0.15, 0.13]
    intra_class_props = [[0.6, 0.4], [0.85, 0.15], [0.75, 0.25], [0.75, 0.25], [0.75, 0.25], [0.85, 0.15]]

    torch.cuda.empty_cache()
    # scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
    if args.mode == "train":
        train_loop(logging.info, cfg,
                   train_criterion, valid_criterion,
                   class_props, intra_class_props, train,
                   AUTOCROP, HFLIP, TRANSPOSE, start_epoch, best_metric)
    '''
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_loader, valid_criterion)
    else:
        submission = test_model(logging.info, cfg, model, test_loader)
        sub_fpath = os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}.csv")
        submission.to_csv(sub_fpath, index=False)
        create_submission(submission, sub_fpath)

    '''


if __name__ == "__main__":
    args = parse_args()
    # print(args)
    cfg = get_cfg()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()
    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)
