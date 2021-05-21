import os
from pprint import pprint

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector


model_cfg = "cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco"

cfg = Config.fromfile(os.path.join("custom_configs", f"{model_cfg}.py"))

# Save configs.
os.makedirs(cfg.work_dir, exist_ok=True)
cfg.dump(os.path.join(cfg.work_dir, "configs.py"))

# model.
model = build_detector(cfg.model)
# dataset.
datasets = [build_dataset(cfg.data.train)]
# training.
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
