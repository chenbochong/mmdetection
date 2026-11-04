import torch
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from .quad_roi_detector import QuadROIDetector


@MODELS.register_module()
class QuadROIDetectorV3(QuadROIDetector):
    def __init__(self, coarse_head, large_height=130, support_iof=0.8, **kwargs):
        super().__init__(**kwargs)
        self.coarse_head = MODELS.build(coarse_head)
        self.large_height = large_height
        self.support_iof = support_iof

    def _ambient_forward(self, inputs):
        p2, p3 = self.extract_feat(inputs)
        output = dict(ambient=self.ambient_head(p2), coarse=self.coarse_head(p3))
        for name, rows in [('ambient', 30), ('coarse', 15)]:
            valid = torch.zeros_like(output[name][:, :1])
            valid[:, :, :rows] = 1
            output[name + '_valid'] = valid
        return p2, output

    def _ambient_losses(self, out, samples):
        unit = out['ambient'].new_tensor([640, 512, 640, 512])
        boxes = [s.gt_instances.bboxes / unit for s in samples]
        ignored = [s.ignored_instances.bboxes / unit for s in samples]
        losses = {'ambient_' + k: v for k, v in self.ambient_head.loss(
            out['ambient'], boxes, ignored, out['ambient_valid']).items()}
        large_boxes, large_ignored = [], []
        for sample, box, ignore in zip(samples, boxes, ignored):
            gt = sample.gt_instances.bboxes
            large = (gt[:, 3] - gt[:, 1]) >= self.large_height
            large_boxes.append(box[large])
            large_ignored.append(torch.cat((ignore, box[~large])))
        losses.update({'ambient_large_' + k: v for k, v in self.coarse_head.loss(
            out['coarse'], large_boxes, large_ignored, out['coarse_valid']).items()})
        return losses

    def _roi_targets(self, sample, roi, affine):
        gt = sample.gt_instances.bboxes
        visible = torch.cat((torch.maximum(gt[:, :2], roi[:2]),
                             torch.minimum(gt[:, 2:], roi[2:])), dim=1)
        area = (visible[:, 2:] - visible[:, :2]).clamp(min=0).prod(dim=1)
        visibility = area / (gt[:, 2:] - gt[:, :2]).prod(dim=1).clamp(min=1)
        selected = visibility >= self.min_visibility
        ignored = torch.cat((sample.ignored_instances.bboxes, gt[~selected]))
        factor, offset = affine[:2].repeat(2), affine[2:].repeat(2)
        full = gt[selected] * factor + offset
        visible = visible[selected] * factor + offset
        points = (visible[:, :2] + visible[:, 2:]) / 2
        return full, visible, ignored * factor + offset, points

    def _roi_losses(self, out, samples):
        targets = [self._roi_targets(samples[int(b)], roi, affine)
                   for b, roi, affine in zip(out['batch'], out['boxes'], out['affine'])]
        full, visible, ignored, points = zip(*targets)
        losses = {'hd_' + k: v for k, v in self.hd_head.loss(
            out['hd'], visible, ignored, out['valid']).items()}
        losses.update({'fused_' + k: v for k, v in self.fused_head.loss(
            out['fused'], full, ignored, out['valid'],
            points=points, heat_boxes=visible).items()})
        return losses

    def _decode_ambient(self, out):
        fine = self.ambient_head.decode(
            out['ambient'], out['ambient_valid'], topk=self.ambient_topk)
        coarse = self.coarse_head.decode(
            out['coarse'], out['coarse_valid'], topk=self.ambient_topk)
        results = [InstanceData.cat([a, b]) for a, b in zip(fine, coarse)]
        for result in results:
            result.bboxes[:, 1::2] *= 512 / 480
        return results

    def _hd_matched_score(self, boxes, evidence, rois, slots):
        roi = rois[torch.searchsorted(slots, evidence.roi_id)]
        visible = torch.cat((torch.maximum(boxes[:, None, :2], roi[None, :, :2]),
                             torch.minimum(boxes[:, None, 2:], roi[None, :, 2:])), dim=2)
        visible_area = (visible[..., 2:] - visible[..., :2]).clamp(min=0).prod(dim=2)
        hd = evidence.bboxes[None]
        intersection = (torch.minimum(visible[..., 2:], hd[..., 2:]) -
                        torch.maximum(visible[..., :2], hd[..., :2])).clamp(min=0).prod(dim=2)
        hd_area = (hd[..., 2:] - hd[..., :2]).clamp(min=0).prod(dim=2)
        full_area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)[:, None]
        iou = intersection / (visible_area + hd_area - intersection).clamp(min=1)
        iof = intersection / hd_area.clamp(min=1)
        coverage = visible_area / full_area.clamp(min=1)
        matches = ((iou >= self.support_iou) & (iof >= self.support_iof) &
                   (coverage >= self.min_visibility))
        return (evidence.scores[None] * matches).max(dim=1)
