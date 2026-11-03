import cv2
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample


@TRANSFORMS.register_module()
class QuadROICrop(BaseTransform):
    def __init__(self, training=True, max_rois=8, crop_scale=(0.6, 1.0),
                 keep_probs=(0.85, 0.65, 0.3), area_thresholds=(4096, 32768),
                 roi_scale=(0.9, 1.4), shift=0.15, random_rois=(0, 3),
                 no_roi_prob=0.15, label_free_prob=0.2, pseudo_rois=True,
                 seed=2026):
        self.training = training
        self.max_rois = max_rois
        self.crop_scale = crop_scale
        self.keep_probs = keep_probs
        self.area_thresholds = area_thresholds
        self.roi_scale = roi_scale
        self.shift = shift
        self.random_rois = random_rois
        self.no_roi_prob = no_roi_prob
        self.label_free_prob = label_free_prob
        self.pseudo_rois = pseudo_rois
        self.seed = seed

    def _vga(self, image, boxes, rng):
        height, width = image.shape[:2]
        if self.training:
            ch = min(height, width * 3 / 4)
            ch = max(4, int(ch * np.sqrt(rng.uniform(*self.crop_scale))))
            cw = max(4, int(ch * 4 / 3))
            x = rng.randint(width - cw + 1)
            y = rng.randint(height - ch + 1)
            image = cv2.resize(image[y:y + ch, x:x + cw], (640, 480))
            scale = np.array([640 / cw, 480 / ch], dtype=np.float32)
            offset = -np.array([x, y], dtype=np.float32) * scale
        else:
            ratio = min(640 / width, 480 / height)
            nw, nh = round(width * ratio), round(height * ratio)
            image_small = cv2.resize(image, (nw, nh))
            x, y = (640 - nw) // 2, (480 - nh) // 2
            image = np.zeros((480, 640, 3), dtype=image.dtype)
            image[y:y + nh, x:x + nw] = image_small
            scale = np.array([nw / width, nh / height], dtype=np.float32)
            offset = np.array([x, y], dtype=np.float32)
        boxes = boxes * np.tile(scale, 2) + np.tile(offset, 2)
        boxes = np.clip(boxes, 0, [640, 480, 640, 480])
        if self.training and rng.rand() < 0.5:
            image = image[:, ::-1].copy()
            boxes[:, [0, 2]] = 640 - boxes[:, [2, 0]]
        return image, boxes, scale, offset

    @staticmethod
    def _align(box):
        lo = np.clip(np.floor(np.asarray(box[:2]) / 4) * 4, 0, [636, 476])
        hi = np.clip(np.ceil(np.asarray(box[2:]) / 4) * 4, lo + 4, [640, 480])
        return np.r_[lo, hi].astype(np.float32)

    def _propose(self, boxes, rng):
        if rng.rand() < self.no_roi_prob:
            return np.empty((0, 4), dtype=np.float32)
        rois = []
        if rng.rand() >= self.label_free_prob:
            for box in boxes:
                size = box[2:] - box[:2]
                group = np.searchsorted(self.area_thresholds, size.prod())
                if rng.rand() >= self.keep_probs[group]:
                    continue
                center = (box[:2] + box[2:]) / 2
                center += rng.uniform(-self.shift, self.shift, 2) * size
                size = size * rng.uniform(*self.roi_scale, size=2)
                rois.append(self._align(np.r_[center - size / 2, center + size / 2]))
        for _ in range(rng.randint(self.random_rois[0], self.random_rois[1] + 1)):
            size = np.array([rng.uniform(20, 180), rng.uniform(32, 280)])
            center = rng.uniform(size / 2, [640, 480] - size / 2)
            rois.append(self._align(np.r_[center - size / 2, center + size / 2]))
        rois = np.unique(np.asarray(rois, dtype=np.float32).reshape(-1, 4), axis=0)
        area = (rois[:, 2:] - rois[:, :2]).prod(axis=1)
        return rois[np.argsort(area, kind='stable')[:self.max_rois]]

    @staticmethod
    def _pack_roi(image, box):
        x1, y1, x2, y2 = box.astype(int)
        block = image[y1:y2, x1:x2]
        # BGR input, RGGB Quad Bayer; channels follow row-major CFA phase.
        phases = [block[y::4, x::4, 2 if y < 2 and x < 2 else
                        0 if y >= 2 and x >= 2 else 1]
                  for y in range(4) for x in range(4)]
        packed = np.stack(phases, axis=-1).astype(np.float32)
        ph, pw = packed.shape[:2]
        ratio = 16 / max(ph, pw)
        nw, nh = max(1, round(pw * ratio)), max(1, round(ph * ratio))
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(packed, (nw, nh), interpolation=interpolation)
        px, py = (16 - nw) // 2, (16 - nh) // 2
        canvas = np.zeros((16, 16, 16), dtype=np.float32)
        canvas[py:py + nh, px:px + nw] = resized
        # VGA xy -> normalized, letterboxed ROI xy.
        scale = np.array([nw / (16 * (x2 - x1)), nh / (16 * (y2 - y1))])
        offset = np.array([px / 16, py / 16]) - box[:2] * scale
        return canvas.transpose(2, 0, 1), np.r_[scale, offset].astype(np.float32)

    def transform(self, results):
        rng = np.random if self.training else np.random.RandomState(
            (self.seed + int(results.get('img_id', 0))) % (2 ** 32))
        image = results['img']
        ori_shape = image.shape[:2]
        boxes = results.get('gt_bboxes', np.empty((0, 4), dtype=np.float32))
        flags = results.get('gt_ignore_flags', np.zeros(len(boxes), dtype=bool))
        image, boxes, scale, offset = self._vga(image, boxes, rng)
        good = ((boxes[:, 2:] - boxes[:, :2]) >= 2).all(axis=1)
        boxes, flags = boxes[good], flags[good]
        if 'roi_boxes' in results:
            # Supplied ROI coordinates are already in the VGA sensor frame.
            rois = np.array([self._align(b) for b in results['roi_boxes']],
                            dtype=np.float32).reshape(-1, 4)
            order = np.argsort((rois[:, 2:] - rois[:, :2]).prod(1), kind='stable')
            rois = rois[order[:self.max_rois]]
        elif self.pseudo_rois:
            rois = self._propose(boxes[~flags.astype(bool)], rng)
        else:
            rois = np.empty((0, 4), dtype=np.float32)
        patches = np.zeros((self.max_rois, 16, 16, 16), dtype=np.float32)
        roi_boxes = np.zeros((self.max_rois, 4), dtype=np.float32)
        affine = np.zeros((self.max_rois, 4), dtype=np.float32)
        valid = np.zeros(self.max_rois, dtype=bool)
        for i, box in enumerate(rois):
            patches[i], affine[i] = self._pack_roi(image, box)
            roi_boxes[i], valid[i] = box, True
        sample = DetDataSample(metainfo=dict(
            img_id=results.get('img_id', 0), img_path=results.get('img_path', ''),
            ori_shape=ori_shape, img_shape=(480, 640),
            vga_scale=scale, vga_offset=offset))
        for name, selected in [('gt_instances', ~flags.astype(bool)),
                               ('ignored_instances', flags.astype(bool))]:
            sample.set_field(InstanceData(
                bboxes=torch.from_numpy(boxes[selected].astype(np.float32)),
                labels=torch.zeros(int(selected.sum()), dtype=torch.long)), name)
        inputs = dict(
            ambient=torch.from_numpy(image[::8, 2::8, 1].copy()[None]).float() / 255,
            rois=torch.from_numpy(patches) / 255,
            roi_boxes=torch.from_numpy(roi_boxes),
            roi_affine=torch.from_numpy(affine),
            roi_valid=torch.from_numpy(valid))
        return dict(inputs=inputs, data_samples=sample)
