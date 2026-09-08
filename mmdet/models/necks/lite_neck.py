from typing import List, Tuple

import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig


@MODELS.register_module()
class LiteNeck(BaseModule):
    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            upsample_cfg: ConfigType = dict(mode='nearest'),
            init_cfg: MultiConfig = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Constant', val=1, layer='BatchNorm2d')
            ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_cfg = upsample_cfg.copy()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(self.in_channels)):
            l_conv = nn.Sequential(
                nn.Conv2d(self.in_channels[i], out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        outs = [self.fpn_convs[i](lateral) for i, lateral in enumerate(laterals)]
        return tuple(outs)