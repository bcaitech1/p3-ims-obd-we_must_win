from pycocotools.coco import COCO
import random
import numpy as np
import torch
import os
import json
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd


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
    def __init__(self, patience=10):
        self._loss_step = 0
        self._miou_step = 0
        self._loss = float('inf')
        self._miou = float('-inf')
        self.patience = patience

    def clear(self):
        self._loss_step = 0
        self._miou_step = 0
        self._loss = float('inf')
        self._miou = float('-inf')

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
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu


def get_mIoU(label_preds, label_trues, n_class):
    batch_miou = []
    for lt, lp in zip(label_trues, label_preds):
        batch_miou += [label_accuracy_score(_fast_hist(lt.flatten(), lp.flatten(), n_class))]
    return batch_miou


def get_scheduler(option, optimizer):
    if option == 0:
        return optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if option == 1:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    if option == 2:
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)


def get_optimizer(model, option, lr):
    if option == 0:
        return optim.AdamW(params=model.parameters(), lr=lr)


def get_criterion():
    return nn.CrossEntropyLoss()


def get_rep(s, label_dis):
    s_list = list(s)
    cur_label_dis = 9999999
    cur_label = 1
    for la in s_list:
        if label_dis[la - 1] < cur_label_dis:
            cur_label_dis = label_dis[la - 1]
            cur_label = la
    return cur_label


def getClassname(class_id, category_list):
    for i in range(len(category_list)):
        if category_list[i]['id'] == class_id:
            return category_list[i]['name']
    return "Background"


def get_dataframe():
    if os.path.exists("/opt/ml/input/data/mydata.bin"):
        with open("/opt/ml/input/data/mydata.bin", "rb") as f:
            df = pickle.load(f)
            return df
    print("make dataframe for Fold")
    label_dis = [160, 2782, 9311, 659, 562, 610, 3090, 1343, 7643, 63, 177]
    fields = ["file_path", "representation_label", "mask"]
    df = pd.DataFrame(columns=fields)
    coco = COCO("/opt/ml/input/data/train_all.json")
    class_dict = make_class_dict(file_name="train_all.json")
    coco_data_num = len(coco.imgs)
    category_id_list = coco.getCatIds()
    category_list = coco.loadCats(category_id_list)
    class_dict = make_class_dict()
    for idx in range(coco_data_num):
        image_id = coco.getImgIds(imgIds=idx)
        image_infos = coco.loadImgs(image_id)[0]  # 1
        file_name = image_infos['file_name']
        ann_id_list = coco.getAnnIds(imgIds=image_infos["id"])
        ann_list = coco.loadAnns(ann_id_list)

        mask = np.zeros((image_infos["height"], image_infos["width"]))
        class_save = set()
        for i in range(len(ann_list)):
            class_name = getClassname(ann_list[i]["category_id"], category_list)
            value = class_dict[class_name]
            class_save.add(value)
            mask = np.maximum(coco.annToMask(ann_list[i]) * value, mask)

        mask = mask.astype(np.float32)  # 2
        rep_label = get_rep(class_save, label_dis)
        df = df.append(pd.Series([file_name, rep_label, mask], index=df.columns), ignore_index=True)
    with open("/opt/ml/input/data/mydata.bin", "wb") as f:
        pickle.dump(df, f)
    print("done make dataframe")
    return df
