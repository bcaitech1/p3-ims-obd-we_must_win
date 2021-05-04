import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.losses import DiceLoss


class DiceLossWithCEE(nn.Module):
    def __init__(self, dice_w=0.5, cee_w=0.5, mode="multiclass", log_loss=False, reduction="mean"):
        super(DiceLossWithCEE, self).__init__()

        self.lambda_1 = dice_w
        self.lambda_2 = cee_w

        self.dice_loss = DiceLoss(mode=mode)
        self.cee_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target):
        dice_out = self.dice_loss(pred, target)
        cee_out = self.cee_loss(pred, target)

        return (self.lambda_1 * dice_out) + (self.lambda_2 * cee_out)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(((1 - prob) ** self.gamma) * log_prob, target_tensor, weight=self.weight, reduction=self.reduction)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
