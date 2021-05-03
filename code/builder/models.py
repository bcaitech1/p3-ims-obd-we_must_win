import torch.nn as nn

from lib.models.seg_hrnet import get_seg_model
from lib.models.seg_hrnet_ocr import get_seg_model as get_seg_ocr_model
from lib.config import config
# from lib.config.hrnet_config import MODEL_CONFIGS


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


# TODO: Not implemented yet.
def HRNet_OCR(cfg_path, pth_path, classes=12):
    config.merge_from_file(cfg_path)
    config.MODEL.PRETRAINED = pth_path
    model = get_seg_ocr_model(config)

    model.ocr_distri_head.cls_head = nn.Conv2d(512, classes, kernel_size=(1, 1), stride=(1, 1))
    model.ocr_distri_head.aux_head[3] = nn.Conv2d(720, classes, kernel_size=(1, 1), stride=(1, 1))

    return model
