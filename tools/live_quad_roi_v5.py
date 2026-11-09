import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from quad_roi_live_utils import RegionMemory, box_state, next_rois, state_box, vga_frame
from quad_roi_runtime import box_area, box_iou, intersection_area
from quad_roi_runtime_v5 import QuadROIV5Onnx


@dataclass
class Track:
    id: int
    state: np.ndarray
    velocity: np.ndarray
    far: bool
    hits: int = 0
    misses: int = 0
    failures: int = 0
    confirmed: bool = False
    stable: int = 0
    ambient_good: int = 0
    ambient_bad: int = 0
    need_roi: bool = True
    last_roi: int = -1000
    score: float = 0
    hd: int = 0


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.memory = RegionMemory(30, 0.04)
        self.tracks = []
        self.frame = 0
        self.next_id = 1

    def update(self, detections, rois, image, now):
        self.frame += 1
        self.memory.update(image, now)
        boxes, scores = detections['bboxes'], detections['scores']
        strong = scores >= self.cfg.confidence
        matches, used = {}, set()
        for track in self.tracks:
            track.state += track.velocity
            track.state[2:] = np.maximum(track.state[2:], 4)
        if self.tracks and len(boxes):
            states = np.stack([t.state for t in self.tracks])
            iou = box_iou(np.stack([state_box(s) for s in states]), boxes)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            drift = np.abs((states[:, None, :2] - centers) / states[:, None, 2:]).mean(2)
            affinity = iou - 0.25 * drift
            for group in (strong, ~strong):
                pairs = np.argwhere((iou >= 0.2) & group[None])
                for ti, di in pairs[np.argsort(-affinity[pairs[:, 0], pairs[:, 1]], kind='stable')]:
                    track = self.tracks[ti]
                    if track.id in matches or di in used:
                        continue
                    matches[track.id] = (di, iou[ti, di])
                    used.add(di)
                    residual = box_state(boxes[di]) - track.state
                    # A weak match can refine position without changing the extrapolation velocity.
                    gain = 1.0 if strong[di] else 0.35
                    track.state += 0.7 * gain * residual
                    track.state[2:] = np.maximum(track.state[2:], 4)
                    if strong[di]:
                        track.velocity += 0.1 * residual
                    scale = np.tile(track.state[2:], 2)
                    stable = strong[di] and np.max(np.abs(residual) / scale) < 0.08 and np.max(np.abs(track.velocity) / scale) < 0.04
                    track.stable = track.stable + 1 if stable else 0
        for di, box in enumerate(boxes):
            if di in used or not strong[di] or self.memory.contains(box):
                continue
            track = Track(self.next_id, box_state(box), np.zeros(4, dtype=np.float32),
                          far=box[3] - box[1] < self.cfg.roi_height)
            self.tracks.append(track)
            matches[track.id] = (di, 1.0)
            self.next_id += 1

        alive = []
        for track in self.tracks:
            di, overlap = matches.get(track.id, (None, 0))
            observed = di is not None and strong[di]
            track.misses = 0 if observed else track.misses + 1
            if di is None:
                track.velocity *= 0.8
                track.stable = 0
            track.score = float(scores[di]) if di is not None else 0
            track.hd = int(detections['hd_support'][di]) if di is not None else 0
            ambient = float(detections['ambient_score'][di]) if di is not None else 0
            track.hits = (track.hits + 1 if overlap >= 0.5 else 1) if observed else 0
            track.confirmed |= track.hits >= 2
            track.ambient_good = track.ambient_good + 1 if ambient >= self.cfg.ambient_high else 0
            track.ambient_bad = track.ambient_bad + 1 if ambient < self.cfg.ambient_low else 0
            if track.ambient_good >= 3:
                track.need_roi = False
            elif track.ambient_bad >= 2:
                track.need_roi = True
            height = track.state[3]
            if height < self.cfg.roi_height - 10:
                track.far = True
            elif height > self.cfg.roi_height + 10:
                track.far = False
            box = np.clip(state_box(track.state), 0, [640, 480, 640, 480]).astype(np.float32)
            if (box[2:] <= box[:2]).any():
                continue
            covered = (intersection_area(box[None], rois)[0] >= 0.8 * box_area(box[None])[0]).any()
            if track.hd or covered:
                track.last_roi = self.frame
            if observed:
                track.failures = 0
            elif covered and not track.confirmed:
                track.failures += 1
            if track.failures >= 2:
                self.memory.remember(box, image, now)
            elif track.misses < self.cfg.max_misses:
                alive.append(track)
        self.tracks = alive

    def requests(self):
        boxes = []
        for track in self.tracks:
            interval = self.cfg.stable_roi_interval if track.far and track.stable >= 3 else 1
            required = not track.confirmed or track.misses or track.need_roi or track.far
            if required and self.frame + 1 - track.last_roi >= interval:
                box = np.clip(state_box(track.state + track.velocity), 0, [640, 480, 640, 480])
                if (box[2:] > box[:2]).all():
                    boxes.append(box)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video')
    parser.add_argument('--log', help='Write per-frame detections, tracks and actual ROIs to JSONL')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--track-confidence', type=float, default=0.10)
    parser.add_argument('--ambient-high', type=float, default=0.30)
    parser.add_argument('--ambient-low', type=float, default=0.20)
    parser.add_argument('--roi-height', type=int, default=130)
    parser.add_argument('--max-misses', type=int, default=5)
    parser.add_argument('--stable-roi-interval', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    cfg = parser.parse_args()
    model, tracker = QuadROIV5Onnx(cfg.model_dir, cfg.threads), Tracker(cfg)
    capture = cv2.VideoCapture(cfg.video if cfg.video else cfg.camera)
    if not cfg.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    rois = np.empty((0, 4), dtype=np.float32)
    log = open(cfg.log, 'w', encoding='utf-8') if cfg.log else None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.perf_counter()
            frame = vga_frame(frame)
            result = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rois,
                                   min(cfg.confidence, cfg.track_confidence))
            now = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if cfg.video else start
            tracker.update(result, rois, frame[::8, 2::8, 1], now)
            used = rois
            rois = next_rois(tracker.requests(), model.settings, 1.15)
            if log:
                entry = dict(frame=tracker.frame, time=now, rois=used.tolist(),
                             detections={k: v.tolist() for k, v in result.items()},
                             tracks=[dict(id=t.id, box=state_box(t.state).tolist(),
                                          confirmed=bool(t.confirmed), misses=t.misses) for t in tracker.tracks])
                log.write(json.dumps(entry) + '\n')
            for roi in used.astype(int):
                cv2.rectangle(frame, tuple(roi[:2]), tuple(roi[2:]), (255, 180, 0), 1)
            for track in tracker.tracks:
                box = np.clip(state_box(track.state), 0, [639, 479, 639, 479]).astype(int)
                status = 'coast' if track.misses else 'confirmed' if track.confirmed else 'pending'
                color = (0, 220, 0) if status == 'confirmed' else (0, 190, 255)
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                cv2.putText(frame, f'#{track.id} {status} S {track.score:.2f} HD {track.hd}',
                            (box[0], max(box[1] - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            ms = (time.perf_counter() - start) * 1000
            cv2.putText(frame, f'{ms:.1f} ms  ROIs {len(used)}/8', (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Quad ROI v5', frame)
            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == '__main__':
    main()
