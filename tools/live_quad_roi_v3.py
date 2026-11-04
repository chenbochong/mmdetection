import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from quad_roi_runtime import QuadROIOnnx, box_iou


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


class AlphaBetaTracker:
    def __init__(self, alpha=0.7, beta=0.1, match_iou=0.2, hd_confidence=0.25):
        self.alpha = alpha
        self.beta = beta
        self.match_iou = match_iou
        self.hd_confidence = hd_confidence
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        boxes, scores = detections['bboxes'], detections['scores']
        matched_tracks, matched_detections = set(), set()
        # Velocity is measured in pixels per processed frame; dt = 1.
        for track in self.tracks:
            track.state += track.velocity
            track.state[2:] = np.maximum(track.state[2:], 4)
        if self.tracks and len(boxes):
            predicted = np.stack([state_box(track.state) for track in self.tracks])
            overlap = box_iou(predicted, boxes)
            pairs = np.argwhere(overlap >= self.match_iou)
            order = np.argsort(-overlap[pairs[:, 0], pairs[:, 1]], kind='stable')
            for ti, di in pairs[order]:
                if ti in matched_tracks or di in matched_detections:
                    continue
                track = self.tracks[ti]
                residual = box_state(boxes[di]) - track.state
                track.state += self.alpha * residual
                track.velocity += self.beta * residual
                track.score = float(scores[di])
                track.confirmed |= bool(detections['hd_support'][di] >= self.hd_confidence)
                track.misses = 0
                matched_tracks.add(ti)
                matched_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.misses += 1
        self.tracks = [track for track in self.tracks if track.misses < 5]
        for di, box in enumerate(boxes):
            if di not in matched_detections:
                self.tracks.append(Track(
                    self.next_id, box_state(box), np.zeros(4, dtype=np.float32),
                    float(scores[di]), bool(detections['hd_support'][di] >= self.hd_confidence)))
                self.next_id += 1


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


def next_rois(tracks, settings, height_limit=200, roi_scale=1.15):
    candidates = []
    for track in tracks:
        if not track.confirmed or track.state[3] <= height_limit or track.misses > 0:
            box = np.clip(state_box(track.state + track.velocity), 0, [640, 480, 640, 480])
            if (box[2:] > box[:2]).all():
                candidates.append(box)
    candidates = np.unique(np.asarray(candidates, dtype=np.float32).reshape(-1, 4), axis=0)
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
    parser.add_argument('--hd-confidence', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--match-iou', type=float, default=0.2)
    parser.add_argument('--roi-height', type=float, default=200)
    parser.add_argument('--roi-scale', type=float, default=1.15)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()

    detector = QuadROIOnnx(args.model_dir, threads=args.threads)
    tracker = AlphaBetaTracker(args.alpha, args.beta, args.match_iou, args.hd_confidence)
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
            result = detector.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rois, args.confidence)
            tracker.update(result)
            for box in rois.astype(int):
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 180, 0), 1)
            for track in tracker.tracks:
                box = np.clip(state_box(track.state), 0, [639, 479, 639, 479]).astype(int)
                color = (0, 180, 255) if track.misses else (0, 220, 0) if track.confirmed else (0, 255, 255)
                status = f'miss {track.misses}/5' if track.misses else 'confirmed' if track.confirmed else 'pending'
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                cv2.putText(frame, f'#{track.id} {status} {track.score:.2f}',
                            (box[0], max(15, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            fps = 1 / (time.perf_counter() - start)
            cv2.putText(frame, f'{fps:.1f} processing FPS  ROIs {len(rois)}/8',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            # Only previous-frame decisions determine this frame's ROI inputs.
            rois = next_rois(tracker.tracks, detector.settings, args.roi_height, args.roi_scale)
            cv2.imshow('Quad ROI v3', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
