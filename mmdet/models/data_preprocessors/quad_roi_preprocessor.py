import torch
from mmengine.model import BaseDataPreprocessor

from mmdet.registry import MODELS


@MODELS.register_module()
class QuadROIPreprocessor(BaseDataPreprocessor):
    def forward(self, data, training=False):
        data = self.cast_data(data)
        inputs = {key: torch.stack(value) if isinstance(value, list) else value
                  for key, value in data['inputs'].items()}
        return dict(inputs=inputs, data_samples=data['data_samples'])


