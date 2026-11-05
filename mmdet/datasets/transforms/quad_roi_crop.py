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
                 seed=2026, height_thresholds=None, merge_cfg=None,
                 min_gt_area=0, partial_prob=0.0, partial_fraction=(0.3, 0.6),
                 matched_negatives=False, negative_attempts=32,
                 hard_negative_slots=0, roi_size=64):
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
        self.height_thresholds = height_thresholds
        self.merge_cfg = merge_cfg
        self.min_gt_area = min_gt_area
        self.partial_prob = partial_prob
        self.partial_fraction = partial_fraction
        self.matched_negatives = matched_negatives
        self.negative_attempts = negative_attempts
        self.hard_negative_slots = hard_negative_slots
        self.packed_size = roi_size // 4

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

    def _jitter(self, box, rng, partial=True):
        size = box[2:] - box[:2]
        center = (box[:2] + box[2:]) / 2
        center += rng.uniform(-self.shift, self.shift, 2) * size
        size = size * rng.uniform(*self.roi_scale, size=2)
        roi = np.r_[center - size / 2, center + size / 2]
        if partial and self.partial_prob and rng.rand() < self.partial_prob:
            axis = rng.randint(2)
            length = size[axis] * rng.uniform(*self.partial_fraction)
            if rng.rand() < 0.5:
                roi[axis + 2] = roi[axis] + length
            else:
                roi[axis] = roi[axis + 2] - length
        return self._align(roi)

    @staticmethod
    def _enclose(boxes):
        return np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]

    @staticmethod
    def _intersects(roi, boxes):
        overlap = np.minimum(roi[2:], boxes[:, 2:]) - np.maximum(roi[:2], boxes[:, :2])
        return (overlap > 0).all(axis=1).any()

    def _background_roi(self, size, occupied, bounds, rng):
        size = np.ceil(size / 4).astype(int) * 4
        lo = np.ceil(bounds[:2] / 4).astype(int) * 4
        hi = np.floor(bounds[2:] / 4).astype(int) * 4
        steps = (hi - lo - size) // 4
        if (steps < 0).any():
            return None
        for _ in range(self.negative_attempts):
            origin = lo + 4 * np.array([rng.randint(n + 1) for n in steps])
            roi = np.r_[origin, origin + size].astype(np.float32)
            if not self._intersects(roi, occupied):
                return roi
        return None

    def _merge(self, rois):
        cfg = self.merge_cfg
        areas = (rois[:, 2:] - rois[:, :2]).prod(axis=1)
        remaining = list(np.argsort(areas, kind='stable'))
        groups = []
        while remaining:
            indices = [remaining.pop(0)]
            while len(indices) < cfg['max_members']:
                best, best_cost = None, float('inf')
                for index in remaining:
                    members = rois[indices + [index]]
                    sizes = members[:, 2:] - members[:, :2]
                    if (sizes.max(axis=0) / sizes.min(axis=0)).max() > cfg['size_ratio']:
                        continue
                    gap = np.maximum(0, np.maximum(members[:-1, :2], members[-1, :2]) -
                                     np.minimum(members[:-1, 2:], members[-1, 2:]))
                    gap = (gap / np.minimum(sizes[:-1], sizes[-1])).max(axis=1).min()
                    if gap > cfg['gap_ratio']:
                        continue
                    extent = self._enclose(members)
                    size = extent[2:] - extent[:2]
                    cost = size.prod() / sizes.prod(axis=1).sum()
                    if cost > cfg['area_ratio']:
                        continue
                    # Allow for the largest jitter and outward 4-pixel alignment.
                    longest = size.max() * max(self.roi_scale) + 8
                    if sizes[:, 1].min() * 64 / longest < cfg['min_height']:
                        continue
                    centers = (members[:, :2] + members[:, 2:]) / 2
                    distance = np.abs(centers[:, None] - centers[None]).max(axis=2)
                    distance = distance[np.triu_indices(len(members), k=1)].min()
                    if distance * 8 / longest < cfg['min_center_distance']:
                        continue
                    if cost < best_cost:
                        best, best_cost = index, cost
                if best is None:
                    break
                indices.append(best)
                remaining.remove(best)
            groups.append(indices)
        return groups

    def _hard_rois(self, boxes, occupied, bounds, rng):
        candidates = [box for box in boxes if not self._intersects(box, occupied)]
        candidates = np.unique(np.asarray(candidates, dtype=np.float32).reshape(-1, 4), axis=0)
        candidates = candidates[rng.permutation(len(candidates))[:self.max_rois]]
        groups = (self._merge(candidates) if self.merge_cfg is not None else
                  [[i] for i in range(len(candidates))])
        rois, counts = [], []
        for group in groups:
            roi = self._jitter(self._enclose(candidates[group]), rng, partial=False)
            outside = (roi[:2] < bounds[:2]).any() or (roi[2:] > bounds[2:]).any()
            # Keep the confusing object in place; never relocate it to easy background.
            if outside or self._intersects(roi, occupied):
                continue
            rois.append(roi)
            counts.append(len(group))
        rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
        area = (rois[:, 2:] - rois[:, :2]).prod(axis=1)
        order = np.lexsort((area, np.asarray(counts) == 1))
        return rois[order[:min(self.hard_negative_slots, self.max_rois)]]

    def _propose(self, boxes, occupied, bounds, rng, hard_boxes):
        if rng.rand() < self.no_roi_prob:
            return np.empty((0, 4), dtype=np.float32)
        rois = []
        if rng.rand() >= self.label_free_prob:
            for box in boxes:
                size = box[2:] - box[:2]
                if self.height_thresholds is None:
                    group = np.searchsorted(self.area_thresholds, size.prod())
                else:
                    group = np.searchsorted(self.height_thresholds, size[1], side='right')
                if rng.rand() >= self.keep_probs[group]:
                    continue
                rois.append(self._jitter(box, rng) if self.merge_cfg is None else box)
        negative = [False] * len(rois)
        templates = np.asarray(rois if rois else boxes, dtype=np.float32).reshape(-1, 4)
        for _ in range(rng.randint(self.random_rois[0], self.random_rois[1] + 1)):
            if self.matched_negatives and len(templates):
                template = templates[rng.randint(len(templates))]
                size = template[2:] - template[:2]
            else:
                size = np.array([rng.uniform(20, 180), rng.uniform(32, 280)])
            if self.matched_negatives:
                roi = self._background_roi(size, occupied, bounds, rng)
                if roi is None:
                    continue
            else:
                center = rng.uniform(size / 2, [640, 480] - size / 2)
                roi = self._align(np.r_[center - size / 2, center + size / 2])
            rois.append(roi)
            negative.append(self.matched_negatives)
        rois, indices = np.unique(
            np.asarray(rois, dtype=np.float32).reshape(-1, 4), axis=0, return_index=True)
        negative = np.asarray(negative, dtype=bool)[indices]
        groups = self._merge(rois) if self.merge_cfg is not None else [[i] for i in range(len(rois))]
        proposals, counts = [], []
        for group in groups:
            roi = (self._jitter(self._enclose(rois[group]), rng)
                   if self.merge_cfg is not None else rois[group[0]])
            if negative[group].all():
                # Merging/jitter can move a background window onto a person or padding.
                outside = (roi[:2] < bounds[:2]).any() or (roi[2:] > bounds[2:]).any()
                if outside or self._intersects(roi, occupied):
                    roi = self._background_roi(roi[2:] - roi[:2], occupied, bounds, rng)
                    if roi is None:
                        continue
            proposals.append(roi)
            counts.append(len(group))
        rois = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
        counts = np.asarray(counts)
        area = (rois[:, 2:] - rois[:, :2]).prod(axis=1)
        order = np.lexsort((area, counts == 1))
        rois = rois[order]
        if self.hard_negative_slots:
            hard = self._hard_rois(hard_boxes, occupied, bounds, rng)
            duplicate = (rois[:, None] == hard[None]).all(axis=2).any(axis=1)
            rois = rois[~duplicate][:self.max_rois - len(hard)]
            return np.concatenate((hard, rois))
        return rois[:self.max_rois]

    def _pack_roi(self, image, box):
        x1, y1, x2, y2 = box.astype(int)
        block = image[y1:y2, x1:x2]
        # BGR input, RGGB Quad Bayer; channels follow row-major CFA phase.
        phases = [block[y::4, x::4, 2 if y < 2 and x < 2 else
                        0 if y >= 2 and x >= 2 else 1]
                  for y in range(4) for x in range(4)]
        packed = np.stack(phases, axis=-1).astype(np.float32)
        ph, pw = packed.shape[:2]
        side = self.packed_size
        ratio = side / max(ph, pw)
        nw, nh = max(1, round(pw * ratio)), max(1, round(ph * ratio))
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        resized = np.stack([
            cv2.resize(packed[..., c], (nw, nh), interpolation=interpolation)
            for c in range(16)], axis=-1)
        px, py = (side - nw) // 2, (side - nh) // 2
        canvas = np.zeros((side, side, 16), dtype=np.float32)
        canvas[py:py + nh, px:px + nw] = resized
        # VGA xy -> normalized, letterboxed ROI xy.
        scale = np.array([nw / (side * (x2 - x1)), nh / (side * (y2 - y1))])
        offset = np.array([px / side, py / side]) - box[:2] * scale
        return canvas.transpose(2, 0, 1), np.r_[scale, offset].astype(np.float32)

    def transform(self, results):
        rng = np.random if self.training else np.random.RandomState(
            (self.seed + int(results.get('img_id', 0))) % (2 ** 32))
        image = results['img']
        ori_shape = image.shape[:2]
        boxes = results.get('gt_bboxes', np.empty((0, 4), dtype=np.float32))
        flags = results.get('gt_ignore_flags', np.zeros(len(boxes), dtype=bool))
        hard_boxes = np.asarray(results.get('hard_negative_boxes', []), dtype=np.float32).reshape(-1, 4)
        count = len(boxes)
        image, all_boxes, scale, offset = self._vga(image, np.concatenate((boxes, hard_boxes)), rng)
        boxes, hard_boxes = all_boxes[:count], all_boxes[count:]
        hard_sizes = hard_boxes[:, 2:] - hard_boxes[:, :2]
        hard_boxes = hard_boxes[(hard_sizes >= 2).all(axis=1) &
                                (hard_sizes.prod(axis=1) >= self.min_gt_area)]
        sizes = boxes[:, 2:] - boxes[:, :2]
        good = (sizes >= 2).all(axis=1) & (sizes.prod(axis=1) >= self.min_gt_area)
        boxes, flags = boxes[good], flags[good]
        if 'roi_boxes' in results:
            # Supplied ROI coordinates are already in the VGA sensor frame.
            rois = np.array([self._align(b) for b in results['roi_boxes']],
                            dtype=np.float32).reshape(-1, 4)
            order = np.argsort((rois[:, 2:] - rois[:, :2]).prod(1), kind='stable')
            rois = rois[order[:self.max_rois]]
        elif self.pseudo_rois:
            bounds = (np.array([0, 0, 640, 480]) if self.training else
                      np.r_[offset, offset + np.rint(scale * ori_shape[::-1])])
            rois = self._propose(boxes[~flags.astype(bool)], boxes, bounds, rng, hard_boxes)
        else:
            rois = np.empty((0, 4), dtype=np.float32)
        patches = np.zeros((self.max_rois, 16, self.packed_size, self.packed_size), dtype=np.float32)
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
