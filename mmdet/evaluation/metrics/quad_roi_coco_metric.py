import numpy as np
from mmengine.logging import MMLogger

from mmdet.datasets.api_wrappers import COCOeval, COCOevalMP
from mmdet.registry import METRICS
from .coco_metric import CocoMetric


def _vga_size(ann, transform):
    scale, offset = transform
    x, y, w, h = ann['bbox']
    box = np.array([x, y, x + w, y + h], dtype=np.float32)
    box = box * np.tile(scale, 2) + np.tile(offset, 2)
    box = np.clip(box, 0, [640, 480, 640, 480])
    return box[2:] - box[:2]


class _HeightCOCOeval(COCOeval):
    def __init__(self, coco_gt, coco_dt, iou_type, transforms, height_thresholds):
        super().__init__(coco_gt, coco_dt, iou_type)
        self.transforms = transforms
        small, large = height_thresholds
        self.params.areaRng = [
            [0, 480], [0, np.nextafter(float(small), -np.inf)],
            [small, np.nextafter(float(large), -np.inf)], [large, 480]]

    def _prepare(self):
        super()._prepare()
        # Reuse COCO's range matching with height, on evaluator-local copies.
        for groups in (self._gts, self._dts):
            for key, anns in groups.items():
                groups[key] = [dict(
                    ann, area=float(_vga_size(ann, self.transforms[ann['image_id']])[1]))
                    for ann in anns]


class _HeightCOCOevalMP(_HeightCOCOeval, COCOevalMP):
    pass


@METRICS.register_module()
class QuadROICocoMetric(CocoMetric):
    """BBox metrics for the Quad ROI detector."""

    def __init__(self, ann_file, min_gt_area=64, height_thresholds=(40, 130),
                 report_ap50_by_size=False, **kwargs):
        super().__init__(ann_file=ann_file, **kwargs)
        self.height_thresholds = height_thresholds
        self.report_ap50_by_size = report_ap50_by_size
        self.vga_transforms = {}
        coco = self._coco_api
        annotations = []
        for image in coco.dataset['images']:
            width, height = image['width'], image['height']
            ratio = min(640 / width, 480 / height)
            nw, nh = round(width * ratio), round(height * ratio)
            scale = np.array([nw / width, nh / height], dtype=np.float32)
            offset = np.array([(640 - nw) // 2, (480 - nh) // 2], dtype=np.float32)
            self.vga_transforms[image['id']] = (scale, offset)
            for ann in coco.imgToAnns[image['id']]:
                size = _vga_size(ann, (scale, offset))
                if (size >= 2).all() and size.prod() >= min_gt_area:
                    annotations.append(ann)
        # Filter GT in VGA units while retaining original COCO coordinates/areas.
        coco.dataset['annotations'] = annotations
        coco.createIndex()
        coco.img_ann_map = coco.imgToAnns
        coco.cat_img_map = coco.catToImgs

    def _create_coco_eval(self, coco_dt, iou_type):
        small, large = self.height_thresholds
        MMLogger.get_current_instance().info(
            f'VGA height groups: small < {small}px, medium [{small}, {large})px, '
            f'large >= {large}px.')
        evaluator = _HeightCOCOevalMP if self.use_mp_eval else _HeightCOCOeval
        return evaluator(self._coco_api, coco_dt, iou_type,
                         self.vga_transforms, self.height_thresholds)

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        self.cat_ids = self._coco_api.get_cat_ids(cat_names=self.dataset_meta['classes'])
        self.img_ids = self._coco_api.get_img_ids()
        preds = [pred for _, pred in results]
        if self.outfile_prefix is not None:
            self.results2json(preds, self.outfile_prefix)
        if self.format_only:
            return {}

        predictions = [dict(
            image_id=pred['img_id'], bbox=self.xyxy2xywh(box),
            score=float(score), category_id=self.cat_ids[label])
            for pred in preds
            for box, score, label in zip(pred['bboxes'], pred['scores'], pred['labels'])]
        logger.info('Evaluating bbox...')
        try:
            coco_dt = self._coco_api.loadRes(predictions)
        except IndexError:
            logger.error('The testing results of the whole dataset is empty.')
            return {}

        evaluator = self._create_coco_eval(coco_dt, 'bbox')
        evaluator.params.catIds = self.cat_ids
        evaluator.params.imgIds = self.img_ids
        evaluator.params.maxDets = list(self.proposal_nums)
        evaluator.params.iouThrs = self.iou_thrs
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        names = ('mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
                 'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000')
        items = self.metric_items if self.metric_items is not None else names[:6]
        logger.info('bbox_mAP_copypaste: ' + ' '.join(
            f'{value:.3f}' for value in evaluator.stats[:6]))
        metrics = {f'bbox_{name}': float(round(evaluator.stats[names.index(name)], 3))
                   for name in items}
        if self.report_ap50_by_size:
            iou50 = np.isclose(evaluator.params.iouThrs, 0.5)
            for area, suffix in enumerate(('', '_s', '_m', '_l')):
                precision = evaluator.eval['precision'][iou50, :, :, area, -1]
                recall = evaluator.eval['recall'][iou50, :, area, -1]
                for name, values in [('mAP_50', precision), ('AR_50', recall)]:
                    values = values[values >= 0]
                    value = float(values.mean()) if values.size else -1.0
                    metrics[f'bbox_{name}{suffix}'] = round(value, 3)
        return metrics
