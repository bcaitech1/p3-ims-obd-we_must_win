import torch.nn as nn

import segmentation_models_pytorch as smp

from torch.optim import Adam, SGD, AdamW
from madgrad import MADGRAD
from adamp import AdamP, SGDP

import cv2
import albumentations as A

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

from builder.criterions import FocalLoss, LabelSmoothingLoss, OhemCrossEntropy
from builder.models import HRNet

model_list = {
    "Unet": smp.Unet,
    "FPN": smp.FPN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
    "PAN": smp.PAN,
    "HRNet": HRNet,
}

criterion_list = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSE": nn.MSELoss,
    "FocalLoss": FocalLoss,
    "KLDiv": nn.KLDivLoss,
    "LabelSmoothingLoss": LabelSmoothingLoss,
    "OhemCrossEntropy": OhemCrossEntropy,
}

optimizer_list = {
    "Adam": Adam,
    "SGD": SGD,
    "MADGRAD": MADGRAD,
    "AdamP": AdamP,
    "SGDP": SGDP,
    "AdamW": AdamW,
}

scheduler_list = {
    "CosineAnnealingLR": CosineAnnealingLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
}

transform_list = {
    "nothing": A.Compose([A.NoOp()]),
    "Resize_256": A.Compose([A.Resize(256, 256, interpolation=cv2.INTER_AREA, always_apply=True)]),
    "Resize_384": A.Compose([A.Resize(384, 384, interpolation=cv2.INTER_AREA, always_apply=True)]),
    "RandomCrop_256": A.Compose([A.RandomCrop(256, 256, always_apply=True)]),
    "RandomCrop_384": A.Compose([A.RandomCrop(384, 384, always_apply=True)]),
    "HorizontalFlip": A.Compose([A.HorizontalFlip(p=1.0)]),
    "VerticalFlip": A.Compose([A.VerticalFlip(p=1.0)]),
    "RandomRotate90": A.Compose([A.RandomRotate90(p=1.0)]),
    "RandomBrightness": A.Compose([A.RandomBrightness(always_apply=True)]),
    "RandomContrast": A.Compose([A.RandomContrast(always_apply=True)]),
}


def Model(model_name, *args, **kwargs):
    if model_name in model_list.keys():
        model = model_list[model_name](*args, **kwargs)
        return model
    else:
        raise Exception(f"{model_name} does not exist in criterion_list.")


def Criterion(criterion_name, *args, **kwargs):
    if criterion_name in criterion_list.keys():
        criterion = criterion_list[criterion_name](*args, **kwargs)
        return criterion
    else:
        raise Exception(f"{criterion_name} does not exist in criterion_list.")


def Optimizer(optimizer_name, *args, **kwargs):
    if optimizer_name in optimizer_list.keys():
        optimizer = optimizer_list[optimizer_name](*args, **kwargs)
        return optimizer
    else:
        raise Exception(f"{optimizer_name} does not exist in optimizer_list.")


def Scheduler(scheduler_name, *args, **kwargs):
    if scheduler_name in scheduler_list.keys():
        scheduler = scheduler_list[scheduler_name](*args, **kwargs)
        return scheduler
    elif scheduler_name == "None":
        return None
    else:
        raise Exception(f"{scheduler_name} does not exist in scheduler_list.")


def Transform(transforms):
    if not transforms:
        return None

    transform_container = []
    for transform_name in transforms:
        if isinstance(transform_name, list) or isinstance(transform_name, tuple):
            list_ = [transform_list[transform_] for transform_ in transform_name]
            transform_container.append(A.Compose(list_))
        else:
            transform_container.append(transform_list[transform_name])
    transform = A.OneOf(transform_container, p=1.0)

    return transform
