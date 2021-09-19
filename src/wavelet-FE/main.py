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
from utils.utils import EarlyStopping, test_time_augmentation
from collections import OrderedDict
from colorama import init, Fore, Back, Style
import matplotlib.pyplot as plt
import json
import torchmetrics
# from torchsampler.imbalanced_sampler import ImbalancedDatasetSampler as imb

warnings.filterwarnings('ignore')
tb = SummaryWriter("runs/wavelet")
metric = torchmetrics.Accuracy(num_classes=6,average='macro')

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
            train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 500))
        data_loader = DataLoader(train_ds, 4, pin_memory=True, shuffle=True, drop_last=False, num_workers=1)

    elif mode == 'valid':
        valid_img_path = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN)
        valid_ds = DataSet(cfg, df, valid_img_path, labels=True, AUTOCROP=autocrop, HFLIP=hflip, TRANSPOSE=transpose,
                           mode="valid")
        if cfg.DEBUG:
            valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 500))
        data_loader = DataLoader(valid_ds, 4, pin_memory=True, shuffle=False, drop_last=False, num_workers=2)
    # test_img_path = os.path.join(cfg.DIRS.DATA,cfg.DIRS.TEST)

    # test_ds = DataSet(cfg,test,test_img_path,labels=True,AUTOCROP=autocrop,HFLIP=hflip,TRANSPOSE=transpose, mode="test")

    return data_loader

def weighted_multi_label_logloss(criterion, prediction, target, weights):
    assert target.size() == prediction.size()
    assert weights.shape[0] == target.size(1)
    loss = 0
    for i in range(target.size(1)):
        loss += weights[i] * criterion(prediction[:, i], target[:, i])
    return loss

def train_loop(_print, cfg, train_criterion, valid_criterion, valid_criterion_2,class_props, intra_class_props, train, AUTOCROP, HFLIP,
               TRANSPOSE, start_epoch, best_metric):
    start = time.time()
    # Create model
    model = build_model(cfg)
    scores, fold_data = [], OrderedDict()
    device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tl = {}
    tc = {}
    acc = {}
    accuracy,total_correct = 0,0
    optimizer = make_optimizer(cfg, model)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

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

    valdf = train[train['fold'] <2].reset_index(drop=True)
    trndf = train[train['fold'] >1].reset_index(drop=True)

    # shuffle the train df
    trndf = shuffle(trndf, random_state=48)
    eno=0
    # split trndf into 6 parts
    #train_csvs = np.array_split(trndf, 3)

    # reset the index of each dataframe
    #train_csvs = [df.reset_index() for df in train_csvs]
    valid_loader = dataloader(cfg, valdf, AUTOCROP, HFLIP, TRANSPOSE, class_props, intra_class_props, mode="valid")
    train_loader = dataloader(cfg, trndf, AUTOCROP, HFLIP, TRANSPOSE, class_props, intra_class_props,mode="train")
    #early_stopping = EarlyStopping(patience=3, verbose=True, file_path=f"{cfg.EXP}.pth")

    #scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS)
    weights = np.array([2, 1, 1, 1, 1, 1])
    weights_norm = weights / weights.sum()
    
    #if torch.cuda.is_available():
            #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    #for k, train in enumerate(train_csvs):
    train_loss, valid_loss, valid_metric = {}, {}, {}

    #train_loader = dataloader(cfg, train, AUTOCROP, HFLIP, TRANSPOSE, class_props, intra_class_props,
                                #mode="train")

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS)
    scheduler = make_lr_scheduler(cfg, optimizer, train_loader)
        
    # Load checkpoint
    try:
        if args.load != "":
            if os.path.isfile(args.load):
                _print(Fore.YELLOW + f"=> loading checkpoint {args.load}")
                checkpoint = torch.load(args.load, "cpu")
                model.load_state_dict(checkpoint.pop('state_dict'))
                if not args.finetune:
                    _print(Fore.GREEN + "resuming optimizer ...")
                    optimizer.load_state_dict(checkpoint.pop('optimizer'))
                    start_epoch, best_metric = checkpoint['epoch']+1, checkpoint['best_metric']
                _print(f"=> loaded checkpoint '{args.load}' (epoch {checkpoint['epoch']}, best_metric: {checkpoint['best_metric']})")
                #train_loss, valid_loss = checkpoint["train_loss"], checkpoint["valid_loss"]
                #valid_metric = checkpoint['valid_metric']
                #optimizer.load_state_dict(checkpoint['optimizer'])
                #scheduler.load_state_dict(checkpoint['lr_scheduler'])
                #start_epoch = checkpoint["epoch"] + 1
                #early_stopping.load_state_dict(checkpoint["stopping_params"])
            else:
                _print(Fore.RED + f"=> no checkpoint found at '{args.load}'")
                start_epoch = 0
    except ValueError:
        _print(Fore.RED + f"=> no checkpoint found at '{args.load}'")
        start_epoch = 0

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):               
        _print(Fore.GREEN + f"Starting Training.....")
        mixed_target = ""
        lamb = ""
        t_loss = 0
        total_loss = 0
        targets_all = []
        outputs_all = []
        t_losses = []
        auc_results=""
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
            t_loss = mixup_criterion(train_criterion, output, target, mixed_target, lamb)
           
            with torch.no_grad():
                #ids_all.extend(ids)
                targets_all.extend(target.cpu().numpy())
                outputs_all.extend(torch.sigmoid(output).cpu().numpy())
                t_losses.append(t_loss.item())
                #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

            #t_loss = weighted_multi_label_logloss(train_criterion,output,target,weights_norm)
            
            #if cfg.SYSTEM.FP16:
                #with amp.scale_loss(t_loss, optimizer) as scaled_loss:
                    #scaled_loss.backward()
            #else:
                #t_loss.backward()
            
            #total_correct += get_num_correct(output, target)
            #accuracy = metric(output.cpu(), target.type(torch.int8).cpu())
            del target, mixed_target, lamb, output
            gc.collect()
            torch.cuda.empty_cache()
            # gradient accumulation
            t_loss = t_loss / cfg.OPT.GD_STEPS
            total_loss+=t_loss.item()
            t_loss.backward()
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
            losses.update(t_loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description(Fore.YELLOW + f"Epoch:({epoch+1}/{2}) ==> Train Loss: {losses.avg:.5f} lr: {optimizer.param_groups[-1]['lr']:.6f}")
            #tbar.set_description(Fore.YELLOW + "Train loss: %.5f, learning rate: %.6f, \
                                #Correct Prediction: %s., Accuracy: %.3f" % (losses.avg, optimizer.param_groups[-1]['lr'],
                                                                                #total_correct,accuracy))
            tb.add_scalar(f"Train_Data/Train_Loss_{epoch+1}", t_loss.item(), x)
            tb.add_scalar(f"Train_Data/Train_Loss_avg_{epoch+1}", losses.avg, x)
            #tb.add_scalar(f"Train_Data/Accuracy_{epoch+1}", accuracy, x)
            
            #tb.add_scalar("Data/Learning Rate", optimizer.param_groups[-1]['lr'], x)
            tb.close()
        result = {
            'targets': np.array(targets_all),
            'outputs': np.array(outputs_all),
            'loss': np.sum(t_losses) / (x+1),
        }
        result.update(calc_auc(result['targets'], result['outputs']))
        result.update(calc_logloss(result['targets'], result['outputs']))    
        
        _print(Fore.GREEN + "Train_loss: %.5f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))
        _print(Fore.GREEN + f"AUC[auc:{result['auc_micro']}, micro:{result['auc_micro']}, macro:{result['auc_macro']}")
        _print(Fore.GREEN + f"Log Loss:{result['logloss']}, Log Loss classes:{np.round(result['logloss_classes'],6)}")
        #_print(Fore.GREEN + "Accuracy: %.3f, Total Correct Prediction: %s." % (accuracy, total_correct))
        
        _print(Fore.YELLOW + f"Training Done in {(time.time() - start) // 60} minutes")
        
        start = time.time()
        _print(Fore.YELLOW + f"Validation Started....")
        auc_score, val_loss = test_time_augmentation(model, valid_loader, cfg, epoch, tb, "Valid",valid_criterion_2,weights_norm,_print,valid_criterion)
        _print(Fore.YELLOW + f"Validation Done in {(time.time() - start) // 60} minutes")
        scores.append(auc_score)
        
        validation_metric = val_loss.item()
        running_loss = total_loss / len(train_loader)
        _print(Fore.GREEN + f'Train Loss: {running_loss:.5f}, '
                            f'Validation Loss: {val_loss:.5f}, '
                            f'Validation Metric: {validation_metric:.5f}, '
                            f'ROC AUC: {auc_score:.5f}')
        is_best = val_loss < best_metric
        best_metric = min(val_loss, best_metric)
        
        _print(Fore.YELLOW + f'Best Metric: {best_metric}')
        
        train_loss[f"train_loss_{epoch}"] = running_loss
        valid_loss[f"valid_loss_{epoch}"] = val_loss.item()
        valid_metric[f"valid_metric_{epoch}"] = validation_metric
       
        if epoch in [2,5,7,10]:
            _print(Fore.YELLOW + 'Saving checkpoint........')
            save_checkpoint({
                "epoch": epoch,
                "arch": cfg.EXP,
                "state_dict": model.state_dict(),
                "best_metric": best_metric,
                "optimizer": optimizer.state_dict(),
            }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")
            
            _print(Fore.YELLOW + 'Checkpoint Saved!')
            
        total_loss,total_correct,accuracy =0,0,0
        '''
        early_stopping(validation_metric, model, epoch=epoch, args=args, optimizer=optimizer.state_dict(),
                        train_loss=train_loss, valid_loss=valid_loss, valid_metric=valid_metric,
                        lr_scheduler=scheduler.state_dict())
        
        if early_stopping.early_stop:
                print("Early stopping")
                break
        '''
        _print(f"Best score: {validation_metric:.5f}")
       
    train_loss_keys = list(train_loss.keys())
    train_loss_values = list(train_loss.values())
    for i in range(len(train_loss_keys)):
        fold_data[train_loss_keys[i]]=pd.Series(train_loss_values[i])
    
    valid_loss_keys = list(valid_loss.keys())
    valid_loss_values = list(valid_loss.values())
    for i in range(len(valid_loss_keys)):
        fold_data[valid_loss_keys[i]]=pd.Series(valid_loss_values[i])
        
    valid_metric_keys = list(valid_metric.keys())
    valid_metric_values = list(valid_metric.values())  
    for i in range(len(valid_metric_keys)):
        fold_data[valid_metric_keys[i]]=pd.Series(valid_metric_values[i])
        
    train_loss = valid_loss = valid_metric = []
        
    print(fold_data)
    metric_values = [v.iloc[-1] for k, v in fold_data.items() if 'valid_metric' in k]
    for i, (ll, roc) in enumerate(zip(metric_values, scores)):
        print(Fore.GREEN + f'Epoch {i+1} LogLoss: {ll:.4f}, ROC AUC: {roc:.4f}')

    print(Fore.CYAN + f'CV mean ROC AUC score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}')
    print(Fore.CYAN + f'CV mean Log Loss: {np.mean(metric_values):.4f}, std: {np.std(metric_values):.4f}')

    '''
    # Make a plot
    fold_data = pd.DataFrame(fold_data)

    fig, ax = plt.subplots()
    plt.title(f"CV Mean score: {np.mean(metric_values):.4f} +/- {np.std(metric_values):.4f}")
    valid_curves = fold_data.loc[:, fold_data.columns.str.startswith('valid_loss')]
    train_curves = fold_data.loc[:, fold_data.columns.str.startswith('train_loss')]
    print(valid_curves,train_curves)
    valid_curves.plot(ax=ax, colormap='Blues_r')
    train_curves.plot(ax=ax, colormap='Reds_r')
    # ax.set_ylim([np.min(train_curves.values), np.max(valid_curves.values)])
    ax.tick_params(labelleft=True, labelright=True, left=True, right=True)
    plt.savefig(os.path.join(cfg.DIRS.OUTPUTS, f"train_epoch_{eno}.png"))
    print("Done in", (time.time() - start) // 60, "minutes")
      
    if valid_loader is not None:
        loss = valid_model(_print, cfg, model, valid_loader, valid_criterion,tb)
        is_best = loss < best_metric
        best_metric = min(loss, best_metric)
        tb.add_scalar("Data/valid_Loss",loss.item(),epoch)
        _print(f"best Metric: {best_metric}")
    
    save_checkpoint({
        "epoch": epoch,
        "arch": cfg.EXP,
        "state_dict": model.state_dict(),
        "best_metric": best_metric,
        "optimizer": optimizer.state_dict(),
    }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")
    '''
def calc_auc(targets, outputs):
    macro = roc_auc_score(np.round(targets), outputs, average='macro')
    micro = roc_auc_score(np.round(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }
    
def calc_logloss(targets, outputs, eps=1e-5):
    # for RSNA
    try:
        logloss_classes = [log_loss(np.round(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    except ValueError as e: 
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
    }
    
def write_json(new_data,details,filename='log.json'):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data[details].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def valid_model(_print, cfg, model, valid_loader, valid_criterion,tb):
    folds = [0, 1, 2, 3, 4]
    # switch to evaluate mode
    model.eval()
    mixed_target = ""
    lamb = ""
    valid_loss= ""
    losses = AverageMeter()
    
    preds = []
    targets = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, batch in enumerate(tbar):
            image = batch['image'].cuda()
            target = batch['labels'].cuda()
            image = rearrange(image, 'b w h c->b c w h')
            '''
            if cfg.DATA.CUTMIX:
                image, target, mixed_target, lamb = cutmix_data(image, target, cfg.DATA.CM_ALPHA)
            elif cfg.DATA.MIXUP:
                image, target, mixed_target, lamb = mixup_data(image, target, cfg.DATA.CM_ALPHA)
            target, mixed_target = target.type(torch.float32), mixed_target.type(torch.float32)
            valid_loss = mixup_criterion(valid_criterion, output, target, mixed_target, lamb)
            '''
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

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()

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
    weights = np.array([2, 1, 1, 1, 1, 1])
    weights_norm = weights / weights.sum()
    
    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(cfg.LOSS.WEIGHTS))
    #train_criterion = nn.BCEWithLogitsLoss()
    valid_criterion = WeightedBCEWithLogitsLoss(class_weights=torch.tensor(cfg.LOSS.WEIGHTS), reduction='none')
    valid_criterion_2 = nn.BCEWithLogitsLoss()

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
                   train_criterion, valid_criterion, valid_criterion_2,
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
