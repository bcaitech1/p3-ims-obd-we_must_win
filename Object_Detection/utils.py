import os
import mmcv


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")


def modify_path(cfg, data_path, save_path):
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = data_path
    cfg.data.train.ann_file = os.path.join(data_path, "train.json")
    #     cfg.data.train.pipeline[2]["img_scale"] = (512, 512)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = data_path
    cfg.data.val.ann_file = os.path.join(data_path, "val.json")
    #     cfg.data.val.pipeline[1]["img_scale"] = (512, 512)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = data_path
    cfg.data.test.ann_file = os.path.join(data_path, "test.json")
    #     cfg.data.test.pipeline[1]["img_scale"] = (512, 512)

    cfg.data.samples_per_gpu = 4

    cfg.seed = 42
    cfg.gpu_ids = [0]
    cfg.work_dir = save_path


def modify_num_classes(cfg, class_num):
    if hasattr(cfg, "num_classes") or "num_classes" in cfg.keys():
        cfg.num_classes = class_num
    for key in cfg:
        child_cfg = getattr(cfg, key)
        if isinstance(child_cfg, mmcv.utils.config.ConfigDict) or isinstance(child_cfg, mmcv.utils.config.Config) or isinstance(child_cfg, dict):
            modify_num_classes(child_cfg, class_num)
        elif isinstance(child_cfg, list):
            for element in child_cfg:
                if isinstance(element, mmcv.utils.config.ConfigDict) or isinstance(element, mmcv.utils.config.Config) or isinstance(element, dict):
                    modify_num_classes(element, class_num)
