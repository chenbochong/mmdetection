import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS


def overlap(a, b):
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1), inter / np.maximum(aa[:, None], 1)


@METRICS.register_module()
class QuadROISceneMetric(BaseMetric):
    default_prefix = 'scene'

    def __init__(self, thresholds=(0.15, 0.25, 0.4), person_threshold=0.7, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = thresholds
        self.person_threshold = person_threshold

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            pred = sample['pred_instances']
            boxes = pred['bboxes'].detach().cpu().numpy()
            boxes = boxes * np.tile(sample['vga_scale'], 2) + np.tile(sample['vga_offset'], 2)
            self.results.append(dict(
                boxes=boxes, scores=pred['scores'].detach().cpu().numpy(),
                person=pred['person_score'].detach().cpu().numpy(),
                gt=sample['gt_instances']['bboxes'].detach().cpu().numpy(),
                ignored=sample['ignored_instances']['bboxes'].detach().cpu().numpy()))

    def compute_metrics(self, results):
        metrics = {}
        for threshold in self.thresholds:
            for verified in (False, True):
                tp = fp = gt_count = small_tp = small_gt = 0
                for row in results:
                    gt = row['gt']
                    small = (gt[:, 3] - gt[:, 1]) < 40
                    gt_count += len(gt)
                    small_gt += int(small.sum())
                    keep = row['scores'] >= threshold
                    if verified:
                        keep &= row['person'] >= self.person_threshold
                    indices = np.flatnonzero(keep)
                    indices = indices[np.argsort(-row['scores'][indices], kind='stable')]
                    used = np.zeros(len(gt), dtype=bool)
                    iou, _ = overlap(row['boxes'][indices], gt)
                    _, ignored_iof = overlap(row['boxes'][indices], row['ignored'])
                    for i in range(len(indices)):
                        options = np.flatnonzero((iou[i] >= 0.5) & ~used)
                        if len(options):
                            best = options[iou[i, options].argmax()]
                            used[best] = True
                            tp += 1
                            small_tp += int(small[best])
                        elif not (ignored_iof[i] >= 0.5).any():
                            fp += 1
                name = f'{"verified" if verified else "detection"}_{threshold:.2f}'
                metrics[name + '_recall50'] = tp / max(gt_count, 1)
                metrics[name + '_small_recall50'] = small_tp / max(small_gt, 1)
                metrics[name + '_fp_per_image'] = fp / max(len(results), 1)
        return metrics
