import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps


@MODELS.register_module()
class QuadROICenterHead(BaseModule):
    def __init__(self, channels=24, direct_regression=False,
                 offset_weight=1.0, wh_weight=0.1, init_cfg=None):
        super().__init__(init_cfg)
        self.direct_regression = direct_regression
        self.offset_weight = offset_weight
        self.wh_weight = wh_weight
        self.output = nn.Conv2d(channels, 6, 1)
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[0] = -4.6
            self.output.bias[1:3] = 1

    def forward(self, x):
        return self.output(x)

    @staticmethod
    def boxes(pred):
        _, _, h, w = pred.shape
        y, x = torch.meshgrid(torch.arange(h, device=pred.device),
                              torch.arange(w, device=pred.device), indexing='ij')
        grid = torch.stack((x, y), dim=0).to(pred.dtype)[None]
        center = grid + pred[:, 3:5].sigmoid()
        size = pred[:, 1:3].relu() + 1e-3
        boxes = torch.cat((center - size / 2, center + size / 2), dim=1)
        return boxes / pred.new_tensor([w, h, w, h])[None, :, None, None]

    def loss(self, pred, gt_boxes, ignored_boxes, valid):
        n, _, h, w = pred.shape
        heat = pred.new_zeros((n, 1, h, w))
        target = pred.new_zeros((n, 4, h, w))
        positive = torch.zeros((n, h, w), device=pred.device, dtype=torch.bool)
        weight = valid.clone()
        scale = pred.new_tensor([w, h, w, h])
        for i, (boxes, ignored) in enumerate(zip(gt_boxes, ignored_boxes)):
            for box in ignored:
                rect = box * scale
                lo = rect[:2].floor().long().clamp(min=0)
                hi = rect[2:].ceil().long()
                weight[i, :, lo[1]:hi[1].clamp(0, h), lo[0]:hi[0].clamp(0, w)] = 0
            area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
            for box in boxes[area.argsort(descending=True)]:
                rect = box * scale
                center = (rect[:2] + rect[2:]) / 2
                x, y = center.floor().long().tolist()
                if not (0 <= x < w and 0 <= y < h) or not valid[i, 0, y, x]:
                    continue
                bw, bh = (rect[2:] - rect[:2]).tolist()
                radius = max(0, int(gaussian_radius((bh, bw), min_overlap=0.3)))
                gen_gaussian_target(heat[i, 0], (x, y), radius)
                target[i, :, y, x] = box
                positive[i, y, x] = True
                weight[i, 0, y, x] = 1
        count = positive.sum().clamp(min=1)
        probability = pred[:, :1].sigmoid()
        pos_loss = F.softplus(-pred[:, :1]) * (1 - probability).square() * heat.eq(1)
        neg_loss = F.softplus(pred[:, :1]) * probability.square() * (1 - heat).pow(4)
        loss_cls = ((pos_loss + neg_loss) * weight).sum() / count
        decoded = self.boxes(pred).permute(0, 2, 3, 1)[positive]
        targets = target.permute(0, 2, 3, 1)[positive]
        iou = bbox_overlaps(decoded, targets, is_aligned=True)
        loss_box = 2 * (1 - iou).sum() / count
        if self.direct_regression:
            grid_boxes = targets * scale
            centers = (grid_boxes[:, :2] + grid_boxes[:, 2:]) / 2
            offsets = centers - centers.floor()
            sizes = grid_boxes[:, 2:] - grid_boxes[:, :2]
            selected = pred.permute(0, 2, 3, 1)[positive]
            loss_box += self.offset_weight * F.l1_loss(
                selected[:, 3:5].sigmoid(), offsets, reduction='sum') / count
            loss_box += self.wh_weight * F.l1_loss(
                selected[:, 1:3].relu() + 1e-3, sizes, reduction='sum') / count
        else:
            loss_box += F.l1_loss(decoded, targets, reduction='sum') / count
        quality = pred[:, 5][positive]
        loss_quality = F.binary_cross_entropy_with_logits(
            quality, iou.detach(), reduction='sum') / count
        return dict(loss_cls=loss_cls, loss_box=loss_box, loss_quality=loss_quality)

    def decode(self, pred, valid, topk=100):
        probability = pred[:, :1].sigmoid()
        peak = probability.eq(F.max_pool2d(probability, 3, stride=1, padding=1))
        scores = probability * pred[:, 5:6].sigmoid() * valid * peak
        values, indices = scores.flatten(1).topk(min(topk, scores[0].numel()), dim=1)
        selected = pred.flatten(2).gather(2, indices[:, None].expand(-1, 6, -1))
        h, w = pred.shape[-2:]
        grid = torch.stack((indices % w, indices // w), dim=1).to(pred.dtype)
        center = grid + selected[:, 3:5].sigmoid()
        size = selected[:, 1:3].relu() + 1e-3
        boxes = torch.cat((center - size / 2, center + size / 2), dim=1)
        boxes = (boxes / pred.new_tensor([w, h, w, h])[None, :, None]).transpose(1, 2)
        return [InstanceData(bboxes=box, scores=value,
                             labels=torch.zeros_like(index))
                for box, index, value in zip(boxes, indices, values)]
