import random
import numpy as np
import torch


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu


def get_mIoU(label_preds, label_trues, n_class):
    batch_mIoU = []
    for lt, lp in zip(label_trues, label_preds):
        batch_mIoU += [label_accuracy_score(_fast_hist(lt.flatten(), lp.flatten(), n_class))]
    return batch_mIoU


# def get_hist(label_preds, label_trues, n_class):
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     return hist


def fix_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
