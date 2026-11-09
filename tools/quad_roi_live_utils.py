from dataclasses import dataclass

import cv2
import numpy as np
from quad_roi_runtime import box_iou


def state_box(state):
    size = np.maximum(state[2:], 4)
    return np.r_[state[:2] - size / 2, state[:2] + size / 2].astype(np.float32)


def box_state(box):
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]].astype(np.float32)


def appearance(ambient, box):
    lo = np.clip(np.floor(box[:2] / 8).astype(int), 0, [79, 59])
    hi = np.clip(np.ceil(box[2:] / 8).astype(int), lo + 1, [80, 60])
    return cv2.resize(ambient[lo[1]:hi[1], lo[0]:hi[0]], (16, 16)).astype(np.float32)


@dataclass
class DeferredRegion:
    box: np.ndarray
    reference: np.ndarray
    expires: float


class RegionMemory:
    def __init__(self, seconds, change_threshold):
        self.seconds = seconds
        self.change_threshold = change_threshold
        self.regions = []

    def update(self, ambient, now):
        kept = []
        for region in self.regions:
            difference = appearance(ambient, region.box) - region.reference
            change = np.abs(difference - np.median(difference)).mean() / 255
            if now < region.expires and change < self.change_threshold:
                kept.append(region)
        self.regions = kept

    def contains(self, box):
        return any(box_iou(box[None], region.box[None])[0, 0] >= 0.4
                   for region in self.regions)

    def remember(self, box, ambient, now):
        self.regions.append(DeferredRegion(box.copy(), appearance(ambient, box), now + self.seconds))
        self.regions = self.regions[-64:]

    def forget(self, box):
        self.regions = [region for region in self.regions
                        if box_iou(box[None], region.box[None])[0, 0] < 0.4]


def enclose(boxes):
    return np.r_[boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)]


def merge_groups(rois, cfg, scale_max):
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
                extent = enclose(members)
                size = extent[2:] - extent[:2]
                cost = size.prod() / sizes.prod(axis=1).sum()
                if cost > cfg['area_ratio']:
                    continue
                longest = size.max() * scale_max + 8
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


def next_rois(candidates, settings, roi_scale=1.15):
    candidates = np.unique(candidates, axis=0)
    groups = merge_groups(candidates, settings['merge_cfg'],
                          max(roi_scale, settings['merge_scale_max']))
    rois, counts = [], []
    for group in groups:
        box = enclose(candidates[group])
        center = (box[:2] + box[2:]) / 2
        size = (box[2:] - box[:2]) * roi_scale
        lo = np.clip(np.floor((center - size / 2) / 4) * 4, 0, [636, 476])
        hi = np.clip(np.ceil((center + size / 2) / 4) * 4, lo + 4, [640, 480])
        rois.append(np.r_[lo, hi])
        counts.append(len(group))
    rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
    area = (rois[:, 2:] - rois[:, :2]).prod(axis=1)
    order = np.lexsort((area, np.asarray(counts) == 1))
    return rois[order[:8]]


def vga_frame(frame):
    h, w = frame.shape[:2]
    if w * 3 > h * 4:
        width = h * 4 // 3
        x = (w - width) // 2
        frame = frame[:, x:x + width]
    else:
        height = w * 3 // 4
        y = (h - height) // 2
        frame = frame[y:y + height]
    return cv2.resize(frame, (640, 480))
