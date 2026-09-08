# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class DualStreamDetector(BaseDetector):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _pack_roi_inputs(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        device = batch_inputs.device
        dtype = batch_inputs.dtype
        roi_list, batch_idx_list, xy_list = [], [], []

        for batch_idx, data_sample in enumerate(batch_data_samples):
            roi = data_sample.roi_img
            xy = data_sample.roi_grid_xy
            if len(roi) == 0:
                continue

            roi = torch.as_tensor(roi, device=device, dtype=dtype).permute(0, 3, 1, 2).contiguous()
            xy = torch.as_tensor(xy, device=device, dtype=torch.long)
            batch_idx_tensor = torch.full((roi.shape[0],), batch_idx, dtype=torch.long, device=device)

            roi_list.append(roi)
            xy_list.append(xy)
            batch_idx_list.append(batch_idx_tensor)

        if len(roi_list) == 0:
            roi_inputs = torch.empty((0, 16, 8, 8), device=device, dtype=dtype)
            roi_grid_xy = torch.empty((0, 2), device=device, dtype=torch.long)
            roi_batch_idx = torch.empty((0,), device=device, dtype=torch.long)
        else:
            roi_inputs = torch.cat(roi_list, dim=0)
            roi_grid_xy = torch.cat(xy_list, dim=0)
            roi_batch_idx = torch.cat(batch_idx_list, dim=0)

        return roi_inputs, roi_batch_idx, roi_grid_xy

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        roi_inputs, roi_batch_idx, roi_grid_xy = self._pack_roi_inputs(batch_inputs, batch_data_samples)
        x = self.extract_feat(batch_inputs, roi_inputs, roi_batch_idx, roi_grid_xy)
        return self.bbox_head.loss(x, batch_data_samples)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        roi_inputs, roi_batch_idx, roi_grid_xy = self._pack_roi_inputs(batch_inputs, batch_data_samples)
        x = self.extract_feat(batch_inputs, roi_inputs, roi_batch_idx, roi_grid_xy)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        roi_inputs, roi_batch_idx, roi_grid_xy = self._pack_roi_inputs(batch_inputs, batch_data_samples)
        x = self.extract_feat(batch_inputs, roi_inputs, roi_batch_idx, roi_grid_xy)
        return self.bbox_head.forward(x)

    def extract_feat(self,
                     batch_inputs: Tensor,
                     roi_inputs: Tensor,
                     roi_batch_idx: Tensor,
                     roi_grid_xy: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs, roi_inputs, roi_batch_idx, roi_grid_xy)
        if self.with_neck:
            x = self.neck(x)
        return x
