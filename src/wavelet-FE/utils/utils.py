import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc
from scipy import interp
from itertools import cycle
from einops import rearrange
import sys
from .augmentation import mixup_data,cutmix_data
#from torch.utils.tensorboard import SummaryWriter
from colorama import Fore
from .logging import AverageMeter
#tbv = SummaryWriter("runs/valid")

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels.argmax(dim=1)).sum().item()

def test_time_augmentation(model, loaders, cfg, epoch, tb, desc,valid_criterion_2,weights,_print=None,valid_criterion=None):
    #tta_predictions = []
    targets = []
    predictions =  []
    tbar = tqdm(loaders,desc=desc)
    with torch.no_grad():
        total_correct=0
        losses = AverageMeter()
        for i, batch in enumerate(tbar):
            image = batch['image'].cuda()
            target = batch['labels']
            if cfg.DATA.CUTMIX:
                image, target, _,_ = cutmix_data(image, target, cfg.DATA.CM_ALPHA)
            elif cfg.DATA.MIXUP:
                image, target,_,_ = mixup_data(image, target, cfg.DATA.CM_ALPHA)
            image = rearrange(image, 'b w h c->b c w h')
            output = model(image)
            target, output = target.type(torch.float32), output.type(torch.float32)
            #vl = weighted_multi_label_logloss(valid_criterion_2,output.cpu(),target.cpu(),weights)
            vl = valid_criterion(output.cpu(), target.cpu())
            vl = vl / cfg.OPT.GD_STEPS
            vl = vl.sum() / valid_criterion.class_weights.sum()
            losses.update(vl.item() * cfg.OPT.GD_STEPS, image.size(0))
            #vl = valid_criterion(output.cpu(),target.cpu())
            #total_correct += get_num_correct(output.cpu(), target.cpu())
            #accuracy = (total_correct/ len(loaders)*100)
            
            tbar.set_description(Fore.GREEN + f"Val for epoch:({epoch+1}) ==> Val Loss: {losses.avg:.5f}")
            tb.add_scalar(f"Val_Data/Val_Loss_{epoch+1}", vl, i)
            #tb.add_scalar(f"Val_Data/Val_Accuracy_{epoch+1}", accuracy, i)
            tb.close()
            predictions.append(output.cpu())
            targets.append(target.cpu())
            
    tta_predictions = torch.cat(predictions, 0)
    predictions = torch.cat(predictions,0)
    targets = torch.cat(targets,0)
    #preds, targets = torch.cat(preds, 0), torch.cat(targets, 0)
    targets = targets.type(torch.float32)
    # record loss
    loss_tensor = valid_criterion(predictions, targets)
    score = roc_auc_score(targets.numpy(), tta_predictions.numpy())
    val_loss = loss_tensor.sum() / valid_criterion.class_weights.sum()
    any_loss = loss_tensor[0]
    intraparenchymal_loss = loss_tensor[1]
    intraventricular_loss = loss_tensor[2]
    subarachnoid_loss = loss_tensor[3]
    subdural_loss = loss_tensor[4]
    epidural_loss = loss_tensor[5]
    _print(
        Fore.LIGHTMAGENTA_EX+"Val. loss: %.5f - any: %.3f - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
            val_loss, any_loss,
            intraparenchymal_loss, intraventricular_loss,
            subarachnoid_loss, subdural_loss, epidural_loss))
    # record AUC
    auc = roc_auc_score(targets[:, 1:].numpy(), predictions[:, 1:].numpy(), average=None)
    _print(
        Fore.LIGHTMAGENTA_EX+"Val. AUC - intraparenchymal: %.3f - intraventricular: %.3f - subarachnoid: %.3f - subdural: %.3f - epidural: %.3f" % (
            auc[0], auc[1], auc[2], auc[3], auc[4]))
    return score,val_loss

def weighted_multi_label_logloss(criterion, prediction, target, weights):
    assert target.size() == prediction.size()
    assert weights.shape[0] == target.size(1)
    loss = 0
    for i in range(target.size(1)):
        loss += weights[i] * criterion(prediction[:, i], target[:, i])
    return loss

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    https://github.com/Bjarten/early-stopping-pytorch
    """
    def __init__(self, patience=7, verbose=False, delta=0, file_path='checkpoint.pt', parallel=False):
        """
        :param patience: How long to wait after last time validation loss improved. Default: 7
        :param verbose: If True, prints a message for each validation loss improvement. Default: False
        :param delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param file_path: Path to save checkpoint file
        :param parallel: If True, the multi-GPU model is saves as a single GPU model
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.file_path = file_path
        self.parallel = parallel
        self.parallel_model = None

    def __call__(self, val_loss, model, **kwargs):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, **kwargs)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, **kwargs)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, **save_items):
        """Saves model and addition items when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if save_items == {}:
            torch.save(model.state_dict(), self.file_path)
        else:
            if self.parallel:
                self.parallel_model = model.state_dict()
                save_items['model'] = model.module.state_dict()
            else:
                save_items['model'] = model.state_dict()
            save_items["stopping_params"] = self.state_dict()
            torch.save(save_items, self.file_path)
        self.val_loss_min = val_loss

    def state_dict(self):
        state = {
            # "counter": self.counter,
            "best_score": self.best_score,
            "val_loss_min": self.val_loss_min,
        }
        return state

    def load_state_dict(self, state):
        # self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.val_loss_min = state["val_loss_min"]


def plot_roc_curve(target, predictions, file_path, metric=None):

    if type(target) == torch.Tensor:
        target = target.numpy()
    if type(predictions) == torch.Tensor:
        predictions = predictions.numpy()

    plt.figure()
    fpr, tpr, _ = roc_curve(target, predictions)
    score = roc_auc_score(target, predictions)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if metric is not None:
        plt.title(f'Receiver operating characteristic. LogLoss: {metric:.4f}')
    else:
        plt.title(f'Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path)


def plot_multiclass_roc_curve(target, predictions, file_path, metric=None):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    if type(target) == torch.Tensor:
        target = target.numpy()
    if type(predictions) == torch.Tensor:
        predictions = predictions.numpy()

    n_classes = target.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    log_loss_vals = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        log_loss_vals[i] = log_loss(target[:, i], predictions[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(target.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'lawngreen', 'gold'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area={1:0.2f}, log_loss={2:0.3f})'
                                                          ''.format(i, roc_auc[i], log_loss_vals[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if metric is not None:
        plt.title(f'Receiver operating characteristic. LogLoss: {metric:.4f}')
    else:
        plt.title(f'Receiver operating characteristic')

    plt.legend(loc="lower right")
    plt.savefig(file_path)


def reindex_submission(df, stage="test1"):
    if stage not in ["test1", "test2"]:
        return df
    elif stage == "test1":
        sub = pd.read_csv(os.path.join(INPUT_DIR, "stage_1_sample_submission.csv"))
    else:
        sub = pd.read_csv(os.path.join(INPUT_DIR, "stage_2_sample_submission.csv"))
    return df.sort_values(by="ID").set_index(sub.sort_values(by="ID").index).sort_index()


def wide_to_long(df, id_var="ImageID"):
    categories = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
    df_long = df.melt(id_vars=[id_var], value_vars=categories, value_name="Label")
    df_long["ID"] = df_long[id_var] + "_" + df_long["variable"]
    df_long = df_long.sort_values(by="ID")
    return df_long[["ID", "Label"]]
