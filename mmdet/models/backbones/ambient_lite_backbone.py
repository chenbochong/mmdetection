# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


class SELayer(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=True),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels // ratio, channels, 1, bias=True),
            nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 with_expand_conv=True,
                 se_ratio=16):
        super().__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        self.with_se = True
        self.with_expand_conv = with_expand_conv
        if not self.with_expand_conv:
            assert mid_channels == in_channels

        if self.with_expand_conv:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True))

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))

        if self.with_se:
            self.se = SELayer(mid_channels, se_ratio)

        self.linear_conv = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = x
        if self.with_expand_conv:
            out = self.expand_conv(out)
        out = self.depthwise_conv(out)
        if self.with_se:
            out = self.se(out)
        out = self.linear_conv(out)
        return x + out if self.with_res_shortcut else out


class RoiSpatialMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


@MODELS.register_module()
class AmbientLiteBackbone(BaseModule):

    def __init__(self,
                 out_indices=(2, 4, 6),
                 fusion_index=2,
                 roi_mixer_blocks=1,
                 roi_scale_init=0.1,
                 global_arch_settings=None,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg)
        if global_arch_settings is None:
            global_arch_settings = [[1, 16, 1, 1, 4], [3, 32, 2, 1, 4], [3, 48, 2, 1, 8],
                                    [3, 64, 2, 2, 8], [3, 64, 2, 1, 16],
                                    [3, 96, 2, 2, 16], [3, 96, 1, 1, 16]]

        self.out_indices = tuple(out_indices)
        self.fusion_index = fusion_index
        self.global_arch_settings = global_arch_settings

        assert fusion_index == 2
        assert all(i >= fusion_index for i in self.out_indices)
        assert global_arch_settings[fusion_index][1] == 48

        self.global_conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.global_layers = []
        in_channels = 16
        for i, layer_cfg in enumerate(self.global_arch_settings[:fusion_index + 1]):
            layer, in_channels = self._make_layer_from_cfg(in_channels, layer_cfg)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            self.global_layers.append(layer_name)

        self.roi_encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True))

        self.roi_mixers = nn.Sequential(*[RoiSpatialMixer(48) for _ in range(roi_mixer_blocks)])
        self.roi_proj = nn.Conv2d(48, 48, kernel_size=1, bias=False)
        self.roi_scale = nn.Parameter(torch.tensor(float(roi_scale_init)))
        self.fuse_act = nn.ReLU(inplace=True)

        self.shared_layers = []
        in_channels = 48
        for i in range(fusion_index + 1, len(self.global_arch_settings)):
            layer, in_channels = self._make_layer_from_cfg(in_channels, self.global_arch_settings[i])
            layer_name = f'shared_layer{i + 1}'
            self.add_module(layer_name, layer)
            self.shared_layers.append((i, layer_name))

        self.output_projs = nn.ModuleDict()
        for i in self.out_indices:
            if i == fusion_index:
                continue
            channels = self.global_arch_settings[i][1]
            self.output_projs[str(i)] = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True))

    def _make_layer_from_cfg(self, in_channels, layer_cfg):
        expand_ratio, out_channels, num_blocks, stride, se_ratio = layer_cfg
        return self.make_layer(in_channels, out_channels, num_blocks, stride, expand_ratio, se_ratio)

    def make_layer(self, in_channels, out_channels, num_blocks, stride, expand_ratio, se_ratio):
        layers = []
        for i in range(num_blocks):
            block_stride = stride if i == 0 else 1
            layers.append(InvertedResidual(
                in_channels,
                out_channels,
                mid_channels=int(round(in_channels * expand_ratio)),
                stride=block_stride,
                with_expand_conv=expand_ratio != 1,
                se_ratio=se_ratio))
            in_channels = out_channels
        return nn.Sequential(*layers), in_channels

    def _scatter_roi(self, roi_feat, roi_batch_idx, roi_grid_xy, batch_size, height, width):
        C = roi_feat.shape[1]
        Hr, Wr = roi_feat.shape[2:]
        dense = roi_feat.new_zeros((batch_size, C, height, width))

        gx = roi_grid_xy[:, 0]
        gy = roi_grid_xy[:, 1]
        yy = torch.arange(Hr, device=roi_feat.device)
        xx = torch.arange(Wr, device=roi_feat.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        ys = gy[:, None, None] * Hr + yy[None, :, :]
        xs = gx[:, None, None] * Wr + xx[None, :, :]
        bs = roi_batch_idx[:, None, None].expand(-1, Hr, Wr)
        indices = (bs * height * width + ys * width + xs).reshape(-1)

        dense_flat = dense.permute(0, 2, 3, 1).reshape(-1, C)
        roi_flat = roi_feat.permute(0, 2, 3, 1).reshape(-1, C)
        dense_flat = dense_flat.index_add(0, indices, roi_flat)
        return dense_flat.reshape(batch_size, height, width, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, roi_x, roi_batch_idx, roi_grid_xy):
        x = self.global_conv1(x)
        for layer_name in self.global_layers:
            x = getattr(self, layer_name)(x)

        if roi_x.shape[0] > 0:
            roi_feat = self.roi_encoder(roi_x)
            roi_dense = self._scatter_roi(roi_feat, roi_batch_idx, roi_grid_xy, x.shape[0], x.shape[2], x.shape[3])
            roi_dense = self.roi_mixers(roi_dense)
            x = x + self.roi_scale * self.roi_proj(roi_dense)

        x = self.fuse_act(x)

        out_map = {}
        if self.fusion_index in self.out_indices:
            out_map[self.fusion_index] = x

        for logical_index, layer_name in self.shared_layers:
            x = getattr(self, layer_name)(x)
            if logical_index in self.out_indices:
                out_map[logical_index] = self.output_projs[str(logical_index)](x)

        return tuple(out_map[i] for i in self.out_indices)
