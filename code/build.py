import torch.nn as nn

import segmentation_models_pytorch as smp

from torch.optim import Adam, SGD, AdamW
from madgrad import MADGRAD
from adamp import AdamP, SGDP

import albumentations as A

from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR

from builder.models import HRNet, SwinTransformerBase
from builder.criterions import FocalLoss, LabelSmoothingLoss, DiceLossWithCEE
from builder.optimizers import RAdam
from builder.transforms import GridMask

model_list = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "FPN": smp.FPN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
    "PAN": smp.PAN,
    "HRNet": HRNet,
    "SwinTransformerBase": SwinTransformerBase
}

criterion_list = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSE": nn.MSELoss,
    "FocalLoss": FocalLoss,
    "KLDiv": nn.KLDivLoss,
    "LabelSmoothingLoss": LabelSmoothingLoss,
    "DiceLossWithCEE": DiceLossWithCEE
}

optimizer_list = {
    "Adam": Adam,
    "SGD": SGD,
    "MADGRAD": MADGRAD,
    "AdamP": AdamP,
    "SGDP": SGDP,
    "AdamW": AdamW,
    "RAdam": RAdam,
}

scheduler_list = {
    "CosineAnnealingLR": CosineAnnealingLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
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
    transform_compose = []
    for transform in transforms:
        if isinstance(transform, list):
            trans_list = [eval(transform_) for transform_ in transform]
            transform_compose.append(A.Compose(trans_list))
        else:
            transform_compose.append(eval(transform))
    transform_compose = A.OneOf(transform_compose)

    return transform_compose
