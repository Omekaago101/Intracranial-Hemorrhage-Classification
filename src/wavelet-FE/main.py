import torch
import torch.nn as nn
from model import WaveletLeTransform
from config.config import get_cfg
from utils import *
from solver import make_lr_scheduler,make_optimizer
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from apex import amp
from sklearn.metrics import roc_auc_score,log_loss
import argparse
import os
from datasets.rsna_dataset import RSNAHemorrhageDS3d
from class_weighted_bce_loss import WeightedBCEWithLogitsLoss
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("--fold", type=int, default=0,
            help="fold for validation")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
        help="model running mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
        help="enable evaluation mode for validation")
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
    model = WaveletLeTransform(cfg.MODEL.WL_CHNS,cfg.MODEL.CONV_CHNS,cfg.MODEL.LEVELS)
    return model

def train_loop(_print, cfg, model, train_loader, criterion, valid_criterion, optimizer, scheduler, start_epoch, best_metric, valid_loader):
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"\nEpoch {epoch + 1}")

        losses = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))

            # calculate loss
            if cfg.DATA.CUTMIX:
                mixed_image, target, mixed_target, lamb = cutmix_data(image, target,
                    cfg.DATA.CM_ALPHA)
            elif cfg.DATA.MIXUP:
                mixed_image, target, mixed_target, lamb = mixup_data(image, target,
                    cfg.DATA.CM_ALPHA)
            output = model(mixed_image, seq_len)
            loss = mixup_criterion(criterion, output,
                target, mixed_target, lamb)

            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS

            if cfg.SYSTEM.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description("Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        _print("Train loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        if valid_loader is not None:
            loss = valid_model(_print, cfg, model, valid_loader, valid_criterion)
            is_best = loss < best_metric
            best_metric = min(loss, best_metric)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": cfg.EXP,
            "state_dict": model.state_dict(),
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")

def valid_model(_print, cfg, model, valid_loader, valid_criterion):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))
            output = model(image, seq_len)
            preds.append(output.cpu())
            targets.append(target.cpu())

    preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    # record loss
    loss_tensor = valid_criterion(preds, targets)
    val_loss = loss_tensor.sum() / valid_criterion.class_weights.sum()
    any_loss = loss_tensor[0]
    intraparenchymal_loss = loss_tensor[1]
    intraventricular_loss = loss_tensor[2]
    subarachnoid_loss = loss_tensor[3]
    subdural_loss = loss_tensor[4]
    epidural_loss = loss_tensor[5]
    _print("Val. loss: %.5f - any: %.3f - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
        val_loss, any_loss,
        intraparenchymal_loss, intraventricular_loss,
        subarachnoid_loss, subdural_loss, epidural_loss))
    # record AUC
    auc = roc_auc_score(targets[:, 1:].numpy(), preds[:, 1:].numpy(), average=None)
    _print("Val. AUC - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
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
    data = [[iid]+[sub_o for sub_o in o] for iid, o in zip(imgid, output)]
    table_data = []
    for subdata in data:
        table_data.append([subdata[0]+'_any', subdata[1]])
        table_data.append([subdata[0]+'_intraparenchymal', subdata[2]])
        table_data.append([subdata[0]+'_intraventricular', subdata[3]])
        table_data.append([subdata[0]+'_subarachnoid', subdata[4]])
        table_data.append([subdata[0]+'_subdural', subdata[5]])
        table_data.append([subdata[0]+'_epidural', subdata[6]])
    df = pd.DataFrame(data=table_data, columns=['ID','Label'])
    df.to_csv(f'{sub_fpath}.csv', index=False)

def main(args, cfg):
    # Set logger
    logging = setup_logger(args.mode, cfg.DIRS.LOGS, 0, filename=f"{cfg.EXP}.txt")

    # Declare variables
    start_epoch = 0
    best_metric = 10.

    # Create model
    model = build_model(cfg)
    
    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.LOSS.WEIGHTS))
    valid_criterion = WeightedBCEWithLogitsLoss(class_weights=torch.tensor(cfg.LOSS.WEIGHTS), reduction='none')
    optimizer = make_optimizer(cfg, model)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()

    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer,
                                          opt_level=cfg.SYSTEM.OPT_L,
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            logging.info(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            model.load_state_dict(ckpt.pop('state_dict'))
            if not args.finetune:
                logging.info("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logging.info(f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    #train = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.TRAIN_CSV ))
    #test = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.TEST_CSV))
    
    DataSet = RSNAHemorrhageDS3d
    train_ds = DataSet(cfg, mode="train")
    valid_ds = DataSet(cfg, mode="valid")
    test_ds = DataSet(cfg, mode="test")
    if cfg.DEBUG:
        train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 50))
        valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 20))

    train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE,
                            pin_memory=True, shuffle=True,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    valid_loader = DataLoader(valid_ds, 1,
                            pin_memory=True, shuffle=True,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    test_loader = DataLoader(test_ds, 1, pin_memory=True, shuffle=False,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
    if args.mode == "train":
        train_loop(logging.info, cfg, model, \
                train_loader, train_criterion, valid_criterion, \
                optimizer, scheduler, start_epoch, best_metric,valid_loader)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_loader, valid_criterion)
    else:
        submission = test_model(logging.info, cfg, model, test_loader)
        sub_fpath = os.path.join(cfg.DIRS.OUTPUTS, f"{cfg.EXP}.csv")
        submission.to_csv(sub_fpath, index=False)
        create_submission(submission, sub_fpath)

    
if __name__ == "__main__":
    args = parse_args()
    print(args)
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