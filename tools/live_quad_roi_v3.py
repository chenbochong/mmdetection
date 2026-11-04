import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from quad_roi_runtime import QuadROIOnnx, box_area, box_iou, intersection_area


def state_box(state):
    size = np.maximum(state[2:], 4)
    return np.r_[state[:2] - size / 2, state[:2] + size / 2].astype(np.float32)


def box_state(box):
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]].astype(np.float32)


@dataclass
class Track:
    id: int
    state: np.ndarray
    velocity: np.ndarray
    score: float
    confirmed: bool
    misses: int = 0
    hd_hits: int = 0
    checks: int = 0
    failed_checks: int = 0
    last_roi: int = -1000
    far: bool = True
    stable_frames: int = 0


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


class AlphaBetaTracker:
    def __init__(self, config):
        self.config = config
        self.memory = RegionMemory(config.memory_seconds, config.memory_change)
        self.tracks = []
        self.next_id = 1
        self.frame = 0

    def update(self, detections, ambient, rois, now):
        cfg = self.config
        self.frame += 1
        self.memory.update(ambient, now)
        boxes, scores = detections['bboxes'], detections['scores']
        strong = scores >= cfg.confidence
        supported = (detections['hd_support'] >= cfg.hd_confidence) & strong
        matched_tracks, matched_detections = set(), set()
        observations = {}
        consistent = set()
        # Velocity is measured in pixels per processed frame; dt = 1.
        for track in self.tracks:
            track.state += track.velocity
            track.state[2:] = np.maximum(track.state[2:], 4)
            track.misses += 1
        if self.tracks and len(boxes):
            predicted = np.stack([state_box(track.state) for track in self.tracks])
            overlap = box_iou(predicted, boxes)
            for selection in (strong, ~strong):
                pairs = np.argwhere((overlap >= cfg.match_iou) & selection[None])
                order = np.argsort(-overlap[pairs[:, 0], pairs[:, 1]], kind='stable')
                for ti, di in pairs[order]:
                    if ti in matched_tracks or di in matched_detections:
                        continue
                    track = self.tracks[ti]
                    residual = box_state(boxes[di]) - track.state
                    relative_error = np.max(np.abs(residual) / np.tile(track.state[2:], 2))
                    gain = 1.0 if strong[di] else 0.5
                    track.state += cfg.alpha * gain * residual
                    track.velocity += cfg.beta * gain * residual
                    track.state[2:] = np.maximum(track.state[2:], 4)
                    speed = np.max(np.abs(track.velocity) / np.tile(track.state[2:], 2))
                    stable = strong[di] and relative_error < 0.08 and speed < 0.04
                    track.stable_frames = track.stable_frames + 1 if stable else 0
                    track.score = float(scores[di])
                    if strong[di]:
                        track.misses = 0
                    observations[track.id] = di
                    if overlap[ti, di] >= 0.5:
                        consistent.add(track.id)
                    matched_tracks.add(ti)
                    matched_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.velocity *= 0.8
                track.stable_frames = 0
        for di, box in enumerate(boxes):
            if di in matched_detections or not strong[di]:
                continue
            if not supported[di] and self.memory.contains(box):
                continue
            self.tracks.append(Track(self.next_id, box_state(box),
                                     np.zeros(4, dtype=np.float32), float(scores[di]), False,
                                     far=box[3] - box[1] < cfg.roi_height))
            observations[self.next_id] = di
            self.next_id += 1

        retained = []
        for track in self.tracks:
            di = observations.get(track.id)
            box = boxes[di] if di is not None else state_box(track.state)
            checked = self.covered(box, rois)
            if checked:
                track.last_roi = self.frame
            if not track.confirmed and (checked or (di is not None and supported[di])):
                track.checks += 1
            if di is not None and supported[di]:
                track.last_roi = self.frame
                track.hd_hits = track.hd_hits + 1 if track.id in consistent else 1
                track.failed_checks = 0
                track.confirmed |= track.hd_hits >= cfg.confirm_hits
                self.memory.forget(box)
            elif checked:
                track.hd_hits = 0
                track.failed_checks += 1
                if track.confirmed and track.failed_checks >= cfg.confirm_attempts:
                    track.confirmed = False
                    track.checks = 0
                    track.failed_checks = 0
                    track.stable_frames = 0
            if track.state[3] < cfg.roi_height - cfg.height_hysteresis:
                track.far = True
            elif track.state[3] > cfg.roi_height + cfg.height_hysteresis:
                track.far = False
            if not track.confirmed and track.checks >= cfg.confirm_attempts:
                self.memory.remember(box, ambient, now)
            elif track.misses < cfg.max_misses:
                retained.append(track)
        self.tracks = retained

    @staticmethod
    def covered(box, rois):
        if not len(rois):
            return False
        coverage = intersection_area(box[None], rois)[0] / max(box_area(box[None])[0], 1)
        projected_height = (box[3] - box[1]) * 64 / (rois[:, 2:] - rois[:, :2]).max(axis=1)
        # Absence of support is meaningful only after an adequately sized, covering ROI.
        return bool(((coverage >= 0.8) & (projected_height >= 24)).any())

    def roi_candidates(self):
        cfg = self.config
        candidates = []
        for track in self.tracks:
            interval = cfg.stable_roi_interval if track.stable_frames >= 3 else 1
            due = track.far and self.frame + 1 - track.last_roi >= interval
            if not track.confirmed or track.misses or due:
                box = np.clip(state_box(track.state + track.velocity), 0, [640, 480, 640, 480])
                if (box[2:] > box[:2]).all():
                    candidates.append(box)
        return np.asarray(candidates, dtype=np.float32).reshape(-1, 4)


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


def main():
    parser = argparse.ArgumentParser(description='Live v3 inference with alpha-beta tracking.')
    parser.add_argument('model_dir')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--track-confidence', type=float, default=0.10)
    parser.add_argument('--hd-confidence', type=float, default=0.25)
    parser.add_argument('--confirm-hits', type=int, default=2)
    parser.add_argument('--confirm-attempts', type=int, default=3)
    parser.add_argument('--memory-seconds', type=float, default=30)
    parser.add_argument('--memory-change', type=float, default=0.04)
    parser.add_argument('--max-misses', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--match-iou', type=float, default=0.2)
    parser.add_argument('--roi-height', type=float, default=130)
    parser.add_argument('--height-hysteresis', type=float, default=10)
    parser.add_argument('--stable-roi-interval', type=int, default=5)
    parser.add_argument('--roi-scale', type=float, default=1.15)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()

    detector = QuadROIOnnx(args.model_dir, threads=args.threads)
    tracker = AlphaBetaTracker(args)
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    rois = np.empty((0, 4), dtype=np.float32)
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            start = time.perf_counter()
            frame = vga_frame(frame)
            ambient = frame[::8, 2::8, 1]
            result = detector.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rois,
                                      min(args.confidence, args.track_confidence))
            tracker.update(result, ambient, rois, time.monotonic())
            used_rois = rois
            rois = next_rois(tracker.roi_candidates(), detector.settings, args.roi_scale)
            for box in used_rois.astype(int):
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 180, 0), 1)
            for track in tracker.tracks:
                box = np.clip(state_box(track.state), 0, [639, 479, 639, 479]).astype(int)
                color = (0, 180, 255) if track.misses else (0, 220, 0) if track.confirmed else (0, 255, 255)
                status = (f'coast {track.misses}/{args.max_misses}' if track.misses else
                          'confirmed' if track.confirmed else 'pending')
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                cv2.putText(frame, f'#{track.id} {status} {track.score:.2f}',
                            (box[0], max(15, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            fps = 1 / (time.perf_counter() - start)
            cv2.putText(frame, f'{fps:.1f} FPS  ROIs {len(used_rois)}/8  memory {len(tracker.memory.regions)}',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow('Quad ROI v3', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
