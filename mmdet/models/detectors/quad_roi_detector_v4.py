import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
from mmengine import MessageHub
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_overlaps
from ..backbones.quad_roi_backbone_v4 import QuadROIBackboneV4
from ..dense_heads.quad_roi_head_v4 import QuadROIHeadV4, QuadROIVerifier
from .quad_roi_detector_v3 import QuadROIDetectorV3


def intersection(a, b):
    return (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
            torch.maximum(a[:, None, :2], b[None, :, :2])).clamp(min=0).prod(-1)


@MODELS.register_module()
class QuadROIDetectorV4(QuadROIDetectorV3):
    def __init__(self, verifier, verifier_warmup=2000, verifier_samples=8,
                 pre_verify_topk=128, **kwargs):
        super().__init__(**kwargs)
        self.verifier = MODELS.build(verifier)
        self.verifier_warmup = verifier_warmup
        self.verifier_samples = verifier_samples
        self.pre_verify_topk = pre_verify_topk

    def _forward(self, inputs, data_samples=None):
        ambient, out = self._ambient_forward(inputs)
        batch, slots = inputs['roi_valid'].nonzero(as_tuple=True)
        out.update(batch=batch, slots=slots)
        if not len(batch):
            return out
        boxes = inputs['roi_boxes'][batch, slots]
        affine = inputs['roi_affine'][batch, slots]
        hd, shallow = self.backbone.roi_encoder(inputs['rois'][batch, slots])
        context, valid = self._align_context(ambient, batch, boxes, affine, hd.shape[-1])
        scale = (boxes[:, 2:] - boxes[:, :2]) / boxes.new_tensor([640, 480])
        fused = self.neck(hd, context, valid, scale)
        out.update(hd=self.hd_head(hd), fused=self.fused_head(fused), shallow=shallow,
                   valid=valid, boxes=boxes, affine=affine)
        return out

    def _verification_loss(self, out, samples):
        crops, targets = [], []
        positive_count = negative_count = hard_count = 0
        step = MessageHub.get_current_instance().get_info('iter') or 0
        with torch.no_grad():
            hd = self.hd_head.decode(out['hd'].detach(), out['valid'], self.roi_topk)
            fused = self.fused_head.decode(out['fused'].detach(), out['valid'], self.roi_topk)
            ambient = self._decode_ambient(out)
        for j, (b, roi, affine) in enumerate(zip(out['batch'], out['boxes'], out['affine'])):
            sample = samples[int(b)]
            _, gt, _, _ = self._roi_targets(sample, roi, affine)
            factor, offset = affine[:2].repeat(2), affine[2:].repeat(2)
            bounds = roi * factor + offset
            occupied = torch.cat((sample.gt_instances.bboxes, sample.ignored_instances.bboxes))
            occupied = occupied * factor + offset
            size = gt[:, 2:] - gt[:, :2]
            jitter = gt + (torch.rand_like(gt) - 0.5) * size.repeat(1, 2) * 0.2
            random = torch.rand((8, 4), device=roi.device)
            lo, hi = torch.minimum(random[:, :2], random[:, 2:]), torch.maximum(random[:, :2], random[:, 2:])
            random = torch.cat((lo, hi), 1) * (bounds[2:] - bounds[:2]).repeat(2) + bounds[:2].repeat(2)
            candidates = [gt, jitter, random, bounds[None]]
            ranks = [roi.new_zeros(sum(len(c) for c in candidates))]
            if step >= self.verifier_warmup:
                a = ambient[int(b)].bboxes.detach() * roi.new_tensor([640, 480, 640, 480])
                candidates += [hd[j].bboxes, fused[j].bboxes, a * factor + offset]
                ranks += [hd[j].scores, fused[j].scores, ambient[int(b)].scores]
            boxes = torch.cat(candidates).detach()
            boxes = torch.cat((torch.maximum(boxes[:, :2], bounds[:2]),
                               torch.minimum(boxes[:, 2:], bounds[2:])), 1)
            enough_pixels = ((boxes[:, 2:] - boxes[:, :2]) * 96 >= roi.new_tensor([4, 12])).all(1)
            boxes = boxes[enough_pixels]
            ranks = torch.cat(ranks).detach()[enough_pixels]
            overlaps = bbox_overlaps(boxes, gt)
            positive = overlaps.max(1).values >= 0.5 if len(gt) else torch.zeros(len(boxes), device=roi.device, dtype=torch.bool)
            negative = ~intersection(boxes, occupied).gt(0).any(1)
            # Ambiguous windows and any unassigned human fragment are not negatives.
            indices, labels = [], []
            for mask, label in ((positive, 1), (negative, 0)):
                selected = mask.nonzero(as_tuple=True)[0]
                hard = selected[:0]
                if label == 0:
                    hard = selected[ranks[selected] >= 0.15]
                    hard = hard[ranks[hard].argsort(descending=True)[:self.verifier_samples // 2]]
                    selected = selected[~torch.isin(selected, hard)]
                    hard_count += len(hard)
                selected = torch.cat((hard, selected[torch.randperm(len(selected), device=roi.device)
                                                    [:self.verifier_samples - len(hard)]]))
                indices.append(selected)
                labels.append(roi.new_full((len(selected),), label))
            selected, label = torch.cat(indices), torch.cat(labels)
            if not len(selected):
                continue
            crops.append(self.verifier.crop(out['shallow'], boxes[selected],
                                           torch.full_like(selected, j)))
            targets.append(label)
            positive_count += int(label.sum())
            negative_count += int((1 - label).sum())
        zero = out['shallow'].sum() * 0 + sum(p.sum() * 0 for p in self.verifier.parameters())
        loss = zero
        if crops:
            logits, target = self.verifier(torch.cat(crops)), torch.cat(targets)
            terms = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            loss = sum(terms[target == label].sum() / (target == label).sum().clamp(min=1)
                       for label in (0, 1)) / 2
        return dict(loss_verify=loss, verify_positive=zero.detach() + positive_count,
                    verify_negative=zero.detach() + negative_count,
                    verify_hard_negative=zero.detach() + hard_count)

    def loss(self, batch_inputs, batch_data_samples):
        out = self._forward(batch_inputs)
        losses = self._ambient_losses(out, batch_data_samples)
        if len(out['batch']):
            losses.update(self._roi_losses(out, batch_data_samples))
            losses.update(self._verification_loss(out, batch_data_samples))
        else:
            modules = (self.backbone.roi_encoder, self.neck, self.hd_head, self.fused_head, self.verifier)
            losses['loss_roi_empty'] = sum(p.sum() * 0 for m in modules for p in m.parameters())
        return losses

    def verify(self, boxes, rois, affine, shallow):
        person = boxes.new_full((len(boxes),), -1)
        coverage = boxes.new_zeros(len(boxes))
        slots = torch.full((len(boxes),), -1, device=boxes.device, dtype=torch.long)
        if not len(boxes) or not len(rois):
            return person, coverage, slots
        lo = torch.maximum(boxes[:, None, :2], rois[None, :, :2])
        hi = torch.minimum(boxes[:, None, 2:], rois[None, :, 2:])
        visible = (hi - lo).clamp(min=0)
        fractions = visible.prod(-1) / (boxes[:, 2:] - boxes[:, :2]).prod(-1)[:, None].clamp(min=1)
        pixels = visible * affine[None, :, :2] * 96
        valid = (fractions >= self.min_visibility) & (pixels >= pixels.new_tensor([4, 12])).all(-1)
        best = (fractions * valid).argmax(1)
        rows = torch.arange(len(boxes), device=boxes.device)
        selected = valid[rows, best]
        index = rows[selected]
        slots[selected] = best[selected]
        coverage[selected] = fractions[index, best[selected]]
        if len(index):
            windows = torch.cat((lo[index, best[selected]], hi[index, best[selected]]), 1)
            transform = affine[best[selected]]
            windows = windows * transform[:, :2].repeat(1, 2) + transform[:, 2:].repeat(1, 2)
            person[selected] = self.verifier(self.verifier.crop(shallow, windows, best[selected])).sigmoid()
        return person, coverage, slots

    def predict(self, inputs, data_samples, rescale=True):
        out = self._forward(inputs)
        ambient = self._decode_ambient(out)
        fused = self.fused_head.decode(out['fused'], out['valid'], self.roi_topk) if len(out['batch']) else []
        results = []
        for b, (a, sample) in enumerate(zip(ambient, data_samples)):
            a.bboxes *= a.bboxes.new_tensor([640, 480, 640, 480])
            a.source = torch.zeros_like(a.labels)
            indices = (out['batch'] == b).nonzero(as_tuple=True)[0]
            candidates = [a]
            for j in indices.tolist():
                f = self._to_vga(fused[j], out['affine'][j])
                f.source = torch.ones_like(f.labels)
                candidates.append(f)
            result = InstanceData.cat(candidates)
            result.bboxes = torch.minimum(result.bboxes.clamp(min=0), result.bboxes.new_tensor([640, 480, 640, 480]))
            keep = (result.center_score >= self.score_thr) & (result.bboxes[:, 2:] > result.bboxes[:, :2]).all(1)
            result = result[keep]
            rank = result.center_score * result.quality_score
            result = result[rank.argsort(descending=True)[:self.pre_verify_topk]]
            if len(indices):
                person, coverage, slots = self.verify(result.bboxes, out['boxes'][indices],
                                                      out['affine'][indices], out['shallow'][indices])
            else:
                person = result.scores.new_full((len(result),), -1)
                coverage = result.scores.new_zeros(len(result))
                slots = torch.full_like(result.labels, -1)
            result.person_score = person
            result.verify_coverage = coverage
            result.roi_id = torch.full_like(result.labels, -1)
            available = slots >= 0
            if len(indices):
                result.roi_id[available] = out['slots'][indices][slots[available]]
            result.scores = result.center_score * torch.where(person >= 0, person, torch.ones_like(person))
            if len(result):
                _, keep = batched_nms(result.bboxes, result.scores * result.quality_score,
                                      result.labels, dict(type='nms', iou_threshold=self.nms_iou))
                result = result[keep[:self.max_per_img]]
            if rescale:
                result.bboxes = (result.bboxes - result.bboxes.new_tensor(sample.vga_offset).repeat(2)) / result.bboxes.new_tensor(sample.vga_scale).repeat(2)
                result.bboxes = torch.minimum(result.bboxes.clamp(min=0), result.bboxes.new_tensor([sample.ori_shape[1], sample.ori_shape[0]] * 2))
            results.append(result)
        return self.add_pred_to_datasample(data_samples, results)
