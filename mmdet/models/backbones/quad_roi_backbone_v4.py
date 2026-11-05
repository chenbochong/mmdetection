import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .quad_roi_backbone import conv, separable


class ROIEncoderV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv(16, 32)
        self.stage2 = nn.Sequential(separable(32, 64, stride=2),
                                    separable(64, 64), separable(64, 64))
        self.stage3 = nn.Sequential(separable(64, 128, stride=2),
                                    separable(128, 128), separable(128, 128))
        self.lateral = conv(64, 128, kernel=1)
        self.refine = separable(128, 128)

    def forward(self, x):
        shallow = self.stem(x)
        fine = self.stage2(shallow)
        deep = self.stage3(fine)
        hd = self.refine(self.lateral(fine) + F.interpolate(deep, scale_factor=2, mode='nearest'))
        return hd, shallow


@MODELS.register_module()
class QuadROIBackboneV4(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.stage2 = nn.Sequential(conv(1, 32, stride=2), separable(32, 64), separable(64, 64))
        self.stage3 = nn.Sequential(separable(64, 96, stride=2), separable(96, 96), separable(96, 96))
        self.stage4 = nn.Sequential(separable(96, 128, stride=2),
                                    *[separable(128, 128) for _ in range(3)])
        self.lateral3 = conv(96, 64, kernel=1)
        self.lateral4 = conv(128, 64, kernel=1)
        self.refine3 = separable(64, 64)
        self.refine2 = separable(64, 64)
        self.roi_encoder = ROIEncoderV4()

    def ambient(self, x):
        a2 = self.stage2(F.pad(x, (0, 0, 0, 4)))
        a3 = self.stage3(a2)
        a4 = self.lateral4(self.stage4(a3))
        p3 = self.refine3(self.lateral3(a3) + F.interpolate(a4, scale_factor=2, mode='nearest'))
        p2 = self.refine2(a2 + F.interpolate(p3, scale_factor=2, mode='nearest'))
        return p2, p3

    def forward(self, x):
        return self.ambient(x)
