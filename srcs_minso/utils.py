import random
import numpy as np
import torch
import os
import json
import pickle


def set_seed(seed: object = 42) -> object:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device() -> object:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"set device : {device}")
    return device


def make_class_dict(data_path="/opt/ml/input/data", file_name="train.json"):
    dict_path = os.path.join(data_path, "classdict.pickle")
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            data = pickle.load(f)
            return data
    with open(os.path.join(data_path, file_name), 'r') as f:
        data = json.loads(f.read())
    class_dict = {"Background": 0}
    categories = data["categories"]
    for category in categories:
        class_dict[category["name"]] = category["id"] + 1
    with open(dict_path, "wb") as f:
        pickle.dump(class_dict, f)
    return class_dict


class EarlyStopping:
    def __init__(self, patience=5):
        self._loss_step = 0
        self._miou_step = 0
        self._loss = float('inf')
        self._miou = float('-inf')
        self.patience = patience

    def validate(self, loss, miou):
        if self._loss < loss:
            self._loss_step += 1
            if self._loss_step > self.patience and self._miou_step > self.patience:
                print(f"stopped early!!")
                return True
        else:
            self._loss_step = 0
            self._loss = loss
        if self._miou > miou:
            self._miou_step += 1
            if self._loss_step > self.patience and self._miou_step > self.patience:
                print(f"stopped early!!")
                return True
        else:
            self._miou_step = 0
            self._miou = miou
        return False


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(hist):
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def add_hist(hist, label_trues, label_preds, n_class):
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    return hist
