import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.utils import reduce_mean
from ..backbones.quad_roi_backbone import conv, separable
from ..dense_heads.fcos_head import FCOSHead
from .base import BaseDetector


class AmbientPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.fine = nn.Sequential(conv(1, 32), separable(32, 32))
        self.medium = nn.Sequential(separable(32, 48, stride=2), separable(48, 48))
        self.coarse = nn.Sequential(separable(48, 96, stride=2), separable(96, 96))
        self.deep = nn.Sequential(separable(96, 128, stride=2), separable(128, 128),
                                  separable(128, 128), conv(128, 32, kernel=1))
        self.lateral2 = conv(48, 32, kernel=1)
        self.lateral3 = conv(96, 32, kernel=1)
        self.refine = nn.ModuleList([separable(32, 32) for _ in range(3)])
        self.presence = nn.Conv2d(32, 1, 1)
        nn.init.constant_(self.presence.bias, -4.6)

    def forward(self, image):
        a1 = self.fine(image)
        a2 = self.medium(a1)
        a3 = self.coarse(a2)
        p3 = self.refine[2](self.lateral3(a3) + F.interpolate(self.deep(a3), size=a3.shape[-2:]))
        p2 = self.refine[1](self.lateral2(a2) + F.interpolate(p3, size=a2.shape[-2:]))
        p1 = self.refine[0](a1 + F.interpolate(p2, size=a1.shape[-2:]))
        levels = (p1, p2, p3)
        return levels, tuple(self.presence(p) for p in levels)


class ROIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv(16, 32)
        self.stage2 = nn.Sequential(separable(32, 64, stride=2),
                                    separable(64, 64), separable(64, 64))
        self.stage3 = nn.Sequential(separable(64, 128, stride=2),
                                    separable(128, 128), separable(128, 128))
        self.lateral = conv(64, 128, kernel=1)
        self.refine = separable(128, 128)

    def forward(self, roi):
        fine = self.stage2(self.stem(roi))
        deep = self.stage3(fine)
        return self.refine(self.lateral(fine) + F.interpolate(deep, scale_factor=2, mode='nearest'))


class ROIFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ROIEncoder()
        self.project = conv(128, 16, kernel=1)

    def forward(self, roi):
        return self.project(self.encoder(roi))


class GlobalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.inject = conv(17, 32, kernel=1)
        self.down2 = separable(32, 32, stride=2)
        self.down3 = separable(32, 32, stride=2)
        self.up2 = separable(32, 32)
        self.up1 = separable(32, 32)

    def forward(self, levels, canvas):
        a1, a2, a3 = levels
        f1 = a1 + self.inject(canvas) * canvas[:, -1:]
        f2 = a2 + self.down2(f1)
        f3 = a3 + self.down3(f2)
        f2 = self.up2(f2 + F.interpolate(f3, size=f2.shape[-2:]))
        f1 = self.up1(f1 + F.interpolate(f2, size=f1.shape[-2:]))
        return f1, f2, f3


class GlobalFCOSHead(FCOSHead):
    def init_weights(self):
        super().init_weights()
        nn.init.constant_(self.conv_reg.bias, 1.0)

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList([separable(self.in_channels, self.feat_channels)])

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList([separable(self.in_channels, self.feat_channels)])

    def losses(self, outputs, ambient, samples):
        cls, dist, ctr = outputs
        points = self.prior_generator.grid_priors([p.shape[-2:] for p in cls], device=cls[0].device)
        labels, targets = self.get_targets(points, [s.gt_instances for s in samples])
        weights = []
        for grid, level_labels in zip(points, labels):
            keep = []
            for sample in samples:
                boxes = sample.ignored_instances.bboxes
                inside = ((grid[:, None] >= boxes[None, :, :2]) &
                          (grid[:, None] < boxes[None, :, 2:])).all(-1).any(1)
                keep.append(~inside)
            weights.append(torch.cat(keep) | (level_labels == 0))
        labels, targets = torch.cat(labels), torch.cat(targets)
        positive = labels == 0
        weight = torch.cat(weights).float()[:, None]
        count = reduce_mean(positive.sum().float()).clamp(min=1)
        logits = torch.cat([p.flatten() for p in cls])[:, None]
        aux = torch.cat([p.flatten() for p in ambient])[:, None]
        losses = dict(loss_person=self.loss_cls(logits, labels, weight=weight, avg_factor=count),
                      loss_ambient=0.5 * self.loss_cls(aux, labels, weight=weight, avg_factor=count))
        predicted = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in dist])[positive]
        target = targets[positive]
        centerness = torch.cat([p.flatten() for p in ctr])[positive]
        quality = self.centerness_target(target)
        denom = reduce_mean(quality.sum().detach()).clamp(min=1e-6)
        if len(target):
            # IoU is translation invariant; distances on both sides are in stride units.
            origin = torch.zeros_like(target[:, :2])
            losses['loss_bbox'] = self.loss_bbox(self.bbox_coder.decode(origin, predicted),
                                                 self.bbox_coder.decode(origin, target),
                                                 weight=quality, avg_factor=denom)
            losses['loss_centerness'] = self.loss_centerness(centerness, quality, avg_factor=count)
        else:
            losses.update(loss_bbox=predicted.sum(), loss_centerness=centerness.sum())
        return losses


@MODELS.register_module()
class QuadROIDetectorV5(BaseDetector):
    def __init__(self, strides, regress_ranges, data_preprocessor, score_thr=0.001,
                 nms_iou=0.5, max_per_img=64, pre_nms=1000):
        super().__init__(data_preprocessor=data_preprocessor)
        self.ambient = AmbientPyramid()
        self.roi = ROIFeatures()
        self.fusion = GlobalFusion()
        self.head = GlobalFCOSHead(num_classes=1, in_channels=32, feat_channels=32,
                                   strides=strides, regress_ranges=regress_ranges,
                                   norm_on_bbox=True, center_sampling=True, centerness_on_reg=True,
                                   loss_bbox=dict(type='GIoULoss', loss_weight=1.0))
        self.score_thr, self.nms_iou = score_thr, nms_iou
        self.max_per_img, self.pre_nms = max_per_img, pre_nms

    def canvas(self, inputs):
        image = inputs['ambient']
        n, _, h, w = image.shape
        batch, slot = inputs['roi_valid'].nonzero(as_tuple=True)
        canvas = image.new_zeros(n, 17, h, w)
        if not len(batch):
            return canvas
        boxes = inputs['roi_boxes'][batch, slot]
        affine = inputs['roi_affine'][batch, slot]
        features = self.roi(inputs['rois'][batch, slot])
        y, x = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device),
                              indexing='ij')
        world = torch.stack((x * 8 + 2.5, y * 8 + 0.5), dim=-1)
        valid = ((world >= boxes[:, None, None, :2]) & (world < boxes[:, None, None, 2:])).all(-1)[:, None]
        uv = world * affine[:, None, None, :2] + affine[:, None, None, 2:]
        # ROI stride-2 feature centers are at (j + 0.25) / 12 on the packed image.
        grid = 2 * (uv + 0.25 / features.shape[-1]) - 1
        aligned = F.grid_sample(features, grid, padding_mode='border', align_corners=False)
        canvas = canvas.index_add(0, batch, torch.cat((aligned * valid, valid.to(aligned.dtype)), dim=1))
        count = canvas[:, -1:]
        return torch.cat((canvas[:, :-1] / count.clamp(min=1), count.clamp(max=1)), dim=1)

    def extract_feat(self, inputs):
        levels, ambient = self.ambient(inputs['ambient'])
        return self.fusion(levels, self.canvas(inputs)), ambient

    def _forward(self, inputs, data_samples=None):
        features, ambient = self.extract_feat(inputs)
        return self.head(features), ambient

    def loss(self, inputs, data_samples):
        predictions, ambient = self._forward(inputs)
        return self.head.losses(predictions, ambient, data_samples)

    def predict(self, inputs, data_samples, rescale=True):
        (cls, dist, ctr), _ = self._forward(inputs)
        points = self.head.prior_generator.grid_priors([p.shape[-2:] for p in cls], device=cls[0].device)
        results = []
        for i, sample in enumerate(data_samples):
            candidates = []
            rois = inputs['roi_boxes'][i, inputs['roi_valid'][i]]
            for c, d, q, grid, (low, high) in zip(cls, dist, ctr, points, self.head.regress_ranges):
                scores = (c[i].flatten().sigmoid() * q[i].flatten().sigmoid()).sqrt()
                indices = torch.where(scores >= self.score_thr)[0]
                indices = indices[scores[indices].argsort(descending=True)[:self.pre_nms]]
                xy = grid[indices]
                sides = d[i].permute(1, 2, 0).reshape(-1, 4)[indices]
                boxes = self.head.bbox_coder.decode(xy, sides, max_shape=(480, 640))
                covered = ((xy[:, None] >= rois[None, :, :2]) & (xy[:, None] < rois[None, :, 2:])).all(-1).any(1)
                distance = sides.amax(1)
                candidates.append(InstanceData(bboxes=boxes, scores=scores[indices],
                                               labels=torch.zeros_like(indices),
                                               hd_support=covered & (distance >= low) & (distance <= high)))
            result = InstanceData.cat(candidates)
            result = result[(result.bboxes[:, 2:] > result.bboxes[:, :2]).all(1)]
            if len(result):
                _, keep = batched_nms(result.bboxes, result.scores, result.labels,
                                      dict(type='nms', iou_threshold=self.nms_iou))
                result = result[keep[:self.max_per_img]]
            if rescale:
                result.bboxes = ((result.bboxes - result.bboxes.new_tensor(sample.vga_offset).repeat(2)) /
                                 result.bboxes.new_tensor(sample.vga_scale).repeat(2))
                result.bboxes = torch.minimum(result.bboxes.clamp(min=0),
                                              result.bboxes.new_tensor(sample.ori_shape[:2][::-1]).repeat(2))
            results.append(result)
        return self.add_pred_to_datasample(data_samples, results)
