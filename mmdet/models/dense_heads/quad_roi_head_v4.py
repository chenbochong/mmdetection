import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from ..backbones.quad_roi_backbone import conv, separable
from .quad_roi_head import QuadROICenterHead


@MODELS.register_module()
class QuadROIHeadV4(QuadROICenterHead):
    def __init__(self, channels, box_mode='center', background_quality_weight=0.1, **kwargs):
        super().__init__(channels=channels, box_mode=box_mode, direct_regression=True, **kwargs)
        del self.tower, self.output
        self.classifier = nn.Sequential(separable(channels, 64), separable(64, 64), nn.Conv2d(64, 1, 1))
        self.regressor = nn.Sequential(separable(channels, 64), separable(64, 64),
                                       nn.Conv2d(64, self.quality_index, 1))
        self.background_quality_weight = background_quality_weight
        for output in (self.classifier[-1], self.regressor[-1]):
            nn.init.normal_(output.weight, std=0.01)
            nn.init.zeros_(output.bias)
        with torch.no_grad():
            self.classifier[-1].bias.fill_(-4.6)
            self.regressor[-1].bias[:self.offset_index - 1] = 1

    def forward(self, x):
        return torch.cat((self.classifier(x), self.regressor(x)), dim=1)

    def loss(self, pred, gt_boxes, ignored_boxes, valid, **kwargs):
        losses = super().loss(pred, gt_boxes, ignored_boxes, valid, **kwargs)
        h, w = pred.shape[-2:]
        penalties = []
        for i, (gt, ignored) in enumerate(zip(gt_boxes, ignored_boxes)):
            background = valid[i, 0].bool().clone()
            for box in torch.cat((gt, ignored)):
                rect = box * box.new_tensor([w, h, w, h])
                lo = rect[:2].floor().long().clamp(min=0)
                hi = rect[2:].ceil().long().clamp(min=0)
                background[lo[1]:hi[1].clamp(max=h), lo[0]:hi[0].clamp(max=w)] = False
            scores = pred[i, 0][background].detach()
            indices = scores.topk(min(16, scores.numel())).indices
            penalties.append(F.softplus(pred[i, self.quality_index][background][indices]).sum() /
                             max(indices.numel(), 1))
        losses['loss_quality_bg'] = self.background_quality_weight * torch.stack(penalties).mean()
        return losses

    def decode(self, pred, valid, topk=100):
        heat = pred[:, :1].sigmoid() * valid
        peaks = heat.eq(F.max_pool2d(heat, 3, 1, 1))
        quality = pred[:, self.quality_index:self.quality_index + 1].sigmoid()
        rank = (heat * quality * peaks).flatten(1)
        _, indices = rank.topk(min(topk, rank.shape[1]), dim=1)
        selected = pred.flatten(2).gather(2, indices[:, None].expand(-1, pred.shape[1], -1))
        h, w = pred.shape[-2:]
        grid = torch.stack((indices % w, indices // w), dim=1).to(pred.dtype)
        boxes = self._decode_boxes(selected, grid).transpose(1, 2) / pred.new_tensor([w, h, w, h])
        centers = (heat * peaks).flatten(1).gather(1, indices)
        qualities = quality.flatten(1).gather(1, indices)
        return [InstanceData(bboxes=b, scores=c, center_score=c, quality_score=q,
                             labels=torch.zeros_like(index))
                for b, c, q, index in zip(boxes, centers, qualities, indices)]


@MODELS.register_module()
class QuadROIVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(conv(32, 64), separable(64, 128, stride=2),
                                    conv(128, 32, kernel=1), nn.Flatten(),
                                    nn.Linear(32 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, crops):
        return self.layers(crops).flatten()

    @staticmethod
    def crop(features, boxes, batch):
        t = (torch.arange(8, device=features.device, dtype=features.dtype) + 0.5) / 8
        y, x = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack((x, y), dim=-1)[None]
        grid = boxes[:, None, None, :2] + xy * (boxes[:, None, None, 2:] - boxes[:, None, None, :2])
        return F.grid_sample(features[batch], 2 * grid - 1, align_corners=False, padding_mode='border')
