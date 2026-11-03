import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..backbones.quad_roi_backbone import conv, separable


@MODELS.register_module()
class QuadROIFusion(BaseModule):
    def __init__(self, ambient_channels=32, roi_channels=64, init_cfg=None):
        super().__init__(init_cfg)
        self.layers = nn.Sequential(
            conv(roi_channels + ambient_channels + 3, roi_channels, kernel=1),
            separable(roi_channels, roi_channels),
            separable(roi_channels, roi_channels))

    def forward(self, hd, ambient, valid, scale):
        scale = scale[:, :, None, None].expand(-1, -1, *hd.shape[-2:])
        return self.layers(torch.cat((hd, ambient, valid, scale), dim=1))
