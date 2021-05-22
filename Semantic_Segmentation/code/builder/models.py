import os

import torch

import torch.nn as nn

from HRNet.models.seg_hrnet import get_seg_model
# from HRNet.models.seg_hrnet_ocr import get_seg_model as get_seg_ocr_model
from HRNet.config import config

from mmcv.utils import Config
from SwinTransformers.mmseg.models import build_segmentor

device = "cuda" if torch.cuda.is_available() else "cpu"


class HRNet(nn.Module):
    def __init__(self, cfg_path, pth_path, classes=12):
        super(HRNet, self).__init__()

        # HRNet
        config.merge_from_file(cfg_path)
        config.MODEL.PRETRAINED = pth_path
        self.hrnet_model = get_seg_model(config)
        del self.hrnet_model.last_layer[3]

        # seg_head
        self.seg_head = nn.Sequential(
            nn.Conv2d(720, classes, kernel_size=(1, 1), stride=(1, 1)),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

    def forward(self, x):
        out = self.hrnet_model(x)
        seg_out = self.seg_head(out)

        return seg_out


class SwinTransformerBase(nn.Module):
    def __init__(self, cfg_path, pth_path):
        super().__init__()
        model_cfg = Config.fromfile(cfg_path)
        model = build_segmentor(model_cfg.model,)
        checkpoints = torch.load(pth_path, map_location=device)

        model.load_state_dict(checkpoints["state_dict"])
        self.model = model
        self.model.decode_head.conv_seg = nn.Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
        self.model.auxiliary_head.conv_seg = nn.Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))

        # swith synbatch to batch ( single gpu training )
        self.synbatch2batch()

    def forward(self, x):
        y = self.model.encode_decode(x, None)
        return y

    def synbatch2batch(self):
        for i in range(len(self.model.decode_head.lateral_convs)):
            synbatch = self.model.decode_head.lateral_convs[i].bn
            batch = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            batch.weight = synbatch.weight
            batch.bias = synbatch.bias
            self.model.decode_head.lateral_convs[i].bn = batch

        for i in range(len(self.model.decode_head.psp_modules)):
            synbatch = self.model.decode_head.psp_modules[i][1].bn
            batch = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            batch.weight = synbatch.weight
            batch.bias = synbatch.bias
            self.model.decode_head.psp_modules[i][1].bn = batch

        for i in range(len(self.model.decode_head.fpn_convs)):
            synbatch = self.model.decode_head.fpn_convs[i].bn
            batch = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            batch.weight = synbatch.weight
            batch.bias = synbatch.bias
            self.model.decode_head.fpn_convs[i].bn = batch

        self.model.decode_head.bottleneck.bn
        synbatch = self.model.decode_head.bottleneck.bn
        batch = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch.weight = synbatch.weight
        batch.bias = synbatch.bias
        self.model.decode_head.bottleneck.bn = batch

        self.model.decode_head.fpn_bottleneck.bn
        synbatch = self.model.decode_head.fpn_bottleneck.bn
        batch = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        batch.weight = synbatch.weight
        batch.bias = synbatch.bias
        self.model.decode_head.fpn_bottleneck.bn = batch
