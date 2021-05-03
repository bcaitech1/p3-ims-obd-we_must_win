import os

import cv2
import numpy as np

import torch

from builder.models import HRNet

model = HRNet("code/builder/seg_hrnet_ocr_w48_520x520_ohem_sgd_lr1e-3_wd1e-4_bs_16_epoch110.yaml", "code/builder/hrnet_ocr_cocostuff_3965_torch04.pth").cuda()
print(model)

test_tensor = torch.randn((2, 3, 512, 512)).cuda()
t_out = model(test_tensor)
print(t_out.shape)
