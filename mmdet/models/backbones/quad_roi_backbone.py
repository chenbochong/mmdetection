import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


def conv(cin, cout, kernel=3, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, kernel // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def separable(cin, cout, stride=1):
    return nn.Sequential(conv(cin, cin, stride=stride, groups=cin),
                         conv(cin, cout, kernel=1))


@MODELS.register_module()
class QuadROIBackbone(BaseModule):
    def __init__(self, ambient_channels=32, roi_channels=64,
                 context_channels=64, init_cfg=None):
        super().__init__(init_cfg)
        self.ambient_stem = nn.Sequential(conv(1, 16, stride=2),
                                          separable(16, ambient_channels))
        self.ambient_context = nn.Sequential(
            separable(ambient_channels, context_channels, stride=2),
            *[separable(context_channels, context_channels) for _ in range(3)],
            conv(context_channels, ambient_channels, kernel=1))
        self.ambient_refine = separable(ambient_channels, ambient_channels)
        self.roi_encoder = nn.Sequential(
            conv(16, roi_channels, stride=2), conv(roi_channels, roi_channels),
            conv(roi_channels, roi_channels),
            separable(roi_channels, roi_channels),
            separable(roi_channels, roi_channels))

    def ambient(self, x):
        fine = self.ambient_stem(x)
        context = self.ambient_context(fine)
        return self.ambient_refine(
            fine + F.interpolate(context, size=fine.shape[-2:], mode='nearest'))

    def roi(self, x):
        return self.roi_encoder(x)

    def forward(self, x):
        return self.ambient(x)


@MODELS.register_module()
class QuadROIBackboneV3(BaseModule):
    def __init__(self, ambient_channels=64, roi_channels=64, init_cfg=None):
        super().__init__(init_cfg)
        c = ambient_channels
        self.ambient_stem = nn.Sequential(conv(1, 16, stride=2), separable(16, c))
        self.ambient_stage3 = nn.Sequential(
            separable(c, c, stride=2), separable(c, c), separable(c, c))
        self.ambient_stage4 = nn.Sequential(
            separable(c, c, stride=2), *[separable(c, c) for _ in range(3)])
        self.ambient_refine3 = separable(c, c)
        self.ambient_refine2 = separable(c, c)
        self.roi_encoder = nn.Sequential(
            conv(16, roi_channels, stride=2), conv(roi_channels, roi_channels),
            conv(roi_channels, roi_channels),
            separable(roi_channels, roi_channels),
            separable(roi_channels, roi_channels))

    def ambient(self, x):
        a2 = self.ambient_stem(F.pad(x, (0, 0, 0, 4)))
        a3 = self.ambient_stage3(a2)
        a4 = self.ambient_stage4(a3)
        p3 = self.ambient_refine3(a3 + F.interpolate(a4, scale_factor=2, mode='nearest'))
        p2 = self.ambient_refine2(a2 + F.interpolate(p3, scale_factor=2, mode='nearest'))
        return p2, p3

    def roi(self, x):
        return self.roi_encoder(x)

    def forward(self, x):
        return self.ambient(x)
