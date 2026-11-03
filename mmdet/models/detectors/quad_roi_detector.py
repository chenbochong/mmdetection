import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps


@MODELS.register_module()
class QuadROIDetector(BaseDetector):
    def __init__(self, backbone, neck, head, ambient_head, data_preprocessor,
                 min_visibility=0.6, score_thr=0.05, nms_iou=0.5,
                 support_iou=0.5, max_per_img=64, ambient_topk=64,
                 roi_topk=16, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.ambient_head = MODELS.build(ambient_head)
        self.hd_head = MODELS.build(head)
        self.fused_head = MODELS.build(head)
        self.min_visibility = min_visibility
        self.score_thr = score_thr
        self.nms_iou = nms_iou
        self.support_iou = support_iou
        self.max_per_img = max_per_img
        self.ambient_topk = ambient_topk
        self.roi_topk = roi_topk

    def extract_feat(self, batch_inputs):
        return self.backbone.ambient(batch_inputs['ambient'])

    @staticmethod
    def _align_context(ambient, batch, boxes, affine, size):
        t = (torch.arange(size, device=ambient.device, dtype=ambient.dtype) + 0.5) / size
        y, x = torch.meshgrid(t, t, indexing='ij')
        xy = torch.stack((x, y), dim=-1)[None]
        world = (xy - affine[:, None, None, 2:]) / affine[:, None, None, :2]
        valid = ((world >= boxes[:, None, None, :2]) &
                 (world < boxes[:, None, None, 2:])).all(dim=-1)[:, None]
        # Ambient samples start at VGA (2, 0); the encoder has stride 2.
        grid = (world - world.new_tensor([2, 0])) / 16
        grid = 2 * (grid + 0.5) / world.new_tensor(
            [ambient.shape[3], ambient.shape[2]]) - 1
        context = F.grid_sample(ambient[batch], grid, align_corners=False,
                                padding_mode='border')
        return context * valid, valid.to(ambient.dtype)

    def _forward(self, inputs, data_samples=None):
        ambient = self.extract_feat(inputs)
        ambient_pred = self.ambient_head(ambient)
        batch, slots = inputs['roi_valid'].nonzero(as_tuple=True)
        output = dict(ambient=ambient_pred, batch=batch, slots=slots)
        if len(batch) == 0:
            return output
        boxes = inputs['roi_boxes'][batch, slots]
        affine = inputs['roi_affine'][batch, slots]
        hd = self.backbone.roi(inputs['rois'][batch, slots])
        context, valid = self._align_context(ambient, batch, boxes, affine, hd.shape[-1])
        scale = (boxes[:, 2:] - boxes[:, :2]) / boxes.new_tensor([640, 480])
        fused = self.neck(hd, context, valid, scale)
        output.update(hd=self.hd_head(hd), fused=self.fused_head(fused),
                      valid=valid, boxes=boxes, affine=affine)
        return output

    def _roi_targets(self, sample, roi, affine):
        gt = sample.gt_instances.bboxes
        intersection = (torch.minimum(gt[:, 2:], roi[2:]) -
                        torch.maximum(gt[:, :2], roi[:2])).clamp(min=0).prod(dim=1)
        visibility = intersection / (gt[:, 2:] - gt[:, :2]).prod(dim=1).clamp(min=1)
        center = (gt[:, :2] + gt[:, 2:]) / 2
        inside = ((center >= roi[:2]) & (center < roi[2:])).all(dim=1)
        selected = inside & (visibility >= self.min_visibility)
        ignored = torch.cat((sample.ignored_instances.bboxes, gt[~selected]))
        factor, offset = affine[:2].repeat(2), affine[2:].repeat(2)
        return gt[selected] * factor + offset, ignored * factor + offset

    def loss(self, batch_inputs, batch_data_samples):
        out = self._forward(batch_inputs)
        pred = out['ambient']
        unit = pred.new_tensor([640, 480, 640, 480])
        boxes = [s.gt_instances.bboxes / unit for s in batch_data_samples]
        ignored = [s.ignored_instances.bboxes / unit for s in batch_data_samples]
        losses = {'ambient_' + k: v for k, v in self.ambient_head.loss(
            pred, boxes, ignored, torch.ones_like(pred[:, :1])).items()}
        if len(out['batch']):
            targets = [self._roi_targets(batch_data_samples[int(b)], roi, affine)
                       for b, roi, affine in zip(out['batch'], out['boxes'], out['affine'])]
            boxes, ignored = zip(*targets)
            for name, head in [('hd', self.hd_head), ('fused', self.fused_head)]:
                losses.update({name + '_' + k: v for k, v in head.loss(
                    out[name], boxes, ignored, out['valid']).items()})
        else:
            # Keep the same distributed graph when a batch contains no ROIs.
            zero = sum(p.sum() * 0 for module in
                       [self.backbone.roi_encoder, self.neck, self.hd_head, self.fused_head]
                       for p in module.parameters())
            for name in ('hd', 'fused'):
                for key in ('loss_cls', 'loss_box', 'loss_quality'):
                    losses[name + '_' + key] = zero
        return losses

    @staticmethod
    def _to_vga(result, affine):
        result.bboxes = (result.bboxes - affine[2:].repeat(2)) / affine[:2].repeat(2)
        return result

    def _matched_score(self, boxes, candidates):
        overlap = bbox_overlaps(boxes, candidates.bboxes)
        scores = candidates.scores[None].expand_as(overlap)
        scores = scores * (overlap >= self.support_iou)
        return scores.max(dim=1)

    def _finish(self, ambient, fused, hd, rois, slots, sample, rescale):
        candidates = InstanceData.cat([ambient] + fused)
        keep = candidates.scores >= self.score_thr
        candidates = candidates[keep]
        candidates.bboxes[:, 0::2].clamp_(0, 640)
        candidates.bboxes[:, 1::2].clamp_(0, 480)
        candidates = candidates[(candidates.bboxes[:, 2:] >
                                  candidates.bboxes[:, :2]).all(dim=1)]
        _, keep = batched_nms(candidates.bboxes, candidates.scores, candidates.labels,
                              dict(type='nms', iou_threshold=self.nms_iou))
        result = candidates[keep[:self.max_per_img]]
        count = len(result)
        result.ambient_score = self._matched_score(result.bboxes, ambient)[0]
        result.hd_support = result.scores.new_zeros(count)
        result.hd_coverage = result.scores.new_zeros(count)
        result.hd_available = torch.zeros(count, device=result.scores.device, dtype=torch.bool)
        result.roi_id = torch.full_like(result.labels, -1)
        if len(rois):
            intersection = (torch.minimum(result.bboxes[:, None, 2:], rois[None, :, 2:]) -
                            torch.maximum(result.bboxes[:, None, :2], rois[None, :, :2]))
            area = (result.bboxes[:, 2:] - result.bboxes[:, :2]).prod(dim=1)
            coverage = intersection.clamp(min=0).prod(dim=2) / area[:, None].clamp(min=1)
            result.hd_coverage, roi_index = coverage.max(dim=1)
            result.hd_available = result.hd_coverage > 0
            result.roi_id = torch.where(result.hd_available, slots[roi_index], -1)
            evidence = InstanceData.cat(hd)
            support, match = self._matched_score(result.bboxes, evidence)
            result.hd_support = support
            result.roi_id = torch.where(support > 0, evidence.roi_id[match], result.roi_id)
        if rescale:
            scale = result.bboxes.new_tensor(sample.vga_scale).repeat(2)
            offset = result.bboxes.new_tensor(sample.vga_offset).repeat(2)
            result.bboxes = (result.bboxes - offset) / scale
            result.bboxes[:, 0::2].clamp_(0, sample.ori_shape[1])
            result.bboxes[:, 1::2].clamp_(0, sample.ori_shape[0])
        return result

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        out = self._forward(batch_inputs)
        ambient = self.ambient_head.decode(
            out['ambient'], torch.ones_like(out['ambient'][:, :1]), topk=self.ambient_topk)
        fused = [[] for _ in ambient]
        hd = [[] for _ in ambient]
        if len(out['batch']):
            fused_decoded = self.fused_head.decode(out['fused'], out['valid'], topk=self.roi_topk)
            hd_decoded = self.hd_head.decode(out['hd'], out['valid'], topk=self.roi_topk)
            for j, (b, slot) in enumerate(zip(out['batch'].tolist(), out['slots'].tolist())):
                f = self._to_vga(fused_decoded[j], out['affine'][j])
                h = self._to_vga(hd_decoded[j], out['affine'][j])
                f.source = torch.ones_like(f.labels)
                h.roi_id = torch.full_like(h.labels, slot)
                fused[b].append(f)
                hd[b].append(h)
        results = []
        for i, (a, sample) in enumerate(zip(ambient, batch_data_samples)):
            a.bboxes *= a.bboxes.new_tensor([640, 480, 640, 480])
            a.source = torch.zeros_like(a.labels)
            slots = batch_inputs['roi_valid'][i].nonzero(as_tuple=True)[0]
            rois = batch_inputs['roi_boxes'][i, slots]
            results.append(self._finish(a, fused[i], hd[i], rois, slots, sample, rescale))
        return self.add_pred_to_datasample(batch_data_samples, results)
