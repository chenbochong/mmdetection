import argparse
import json
import time
from dataclasses import dataclass

import cv2
import numpy as np
from live_quad_roi_v3 import RegionMemory, box_state, next_rois, state_box, vga_frame
from quad_roi_runtime import box_iou
from quad_roi_runtime_v4 import QuadROIV4Onnx


@dataclass
class Track:
    id: int
    state: np.ndarray
    velocity: np.ndarray
    confirmed: bool = False
    hits: int = 0
    negatives: int = 0
    misses: int = 0
    ambient_good: int = 0
    ambient_bad: int = 0
    need_roi: bool = True
    far: bool = True
    stable: int = 0
    uncertain: int = 0
    last_roi: int = -1000
    observed: int = 0
    text: tuple = ()


class Tracker:
    def __init__(self, args):
        self.args = args
        self.memory = RegionMemory(args.memory_seconds, args.memory_change)
        self.tracks = []
        self.frame = 0
        self.next_id = 1

    def update(self, result, ambient, now):
        self.frame += 1
        cfg = self.args
        self.memory.update(ambient, now)
        boxes, centers = result['bboxes'], result['center_score']
        strong = centers >= cfg.confidence
        matched_tracks, matched_detections, observations, consistent = set(), set(), {}, set()
        for track in self.tracks:
            track.state += track.velocity
            track.state[2:] = np.maximum(track.state[2:], 4)
            track.misses += 1
        if self.tracks and len(boxes):
            overlaps = box_iou(np.stack([state_box(t.state) for t in self.tracks]), boxes)
            for mask in (strong, ~strong):
                pairs = np.argwhere((overlaps >= cfg.match_iou) & mask[None])
                order = np.argsort(-overlaps[pairs[:, 0], pairs[:, 1]], kind='stable')
                for ti, di in pairs[order]:
                    if ti in matched_tracks or di in matched_detections:
                        continue
                    track = self.tracks[ti]
                    error = box_state(boxes[di]) - track.state
                    relative = np.max(np.abs(error) / np.tile(track.state[2:], 2))
                    gain = float(np.clip(result['quality_score'][di], 0.2, 1))
                    track.state += cfg.alpha * gain * error
                    track.velocity += cfg.beta * gain * error
                    track.state[2:] = np.maximum(track.state[2:], 4)
                    speed = np.max(np.abs(track.velocity) / np.tile(track.state[2:], 2))
                    stable = strong[di] and relative < 0.08 and speed < 0.04
                    track.stable = track.stable + 1 if stable else 0
                    observations[track.id] = di
                    if overlaps[ti, di] >= 0.5:
                        consistent.add(track.id)
                    matched_tracks.add(ti)
                    matched_detections.add(di)
        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                track.velocity *= 0.8
                track.stable = 0
        for di, box in enumerate(boxes):
            if di in matched_detections or not strong[di]:
                continue
            if result['person_score'][di] < cfg.person_high and self.memory.contains(box):
                continue
            self.tracks.append(Track(self.next_id, box_state(box), np.zeros(4, dtype=np.float32),
                                     far=box[3] - box[1] < cfg.roi_height))
            observations[self.next_id] = di
            self.next_id += 1
        retained = []
        for track in self.tracks:
            di = observations.get(track.id)
            ac = aq = 0
            negative = False
            if di is not None:
                person = result['person_score'][di]
                coverage = result['verify_coverage'][di]
                positive = person >= cfg.person_high
                negative = 0 <= person <= cfg.person_low and coverage >= 0.8
                if strong[di] and not negative:
                    track.misses = 0
                ac, aq = result['ambient_center'][di], result['ambient_quality'][di]
                if person >= 0:
                    track.last_roi = self.frame
                    if positive:
                        track.hits = track.hits + 1 if track.id in consistent else 1
                        track.confirmed |= track.hits >= cfg.confirm_hits
                        track.negatives = track.uncertain = 0
                        self.memory.forget(boxes[di])
                    elif negative:
                        track.hits = 0
                        track.negatives += 1
                        track.uncertain = 0
                    else:
                        track.hits = track.negatives = 0
                        track.uncertain += 1
                source = 'A' if result['source'][di] == 0 else 'F'
                track.text = (f"{source} C {centers[di]:.2f} Q {result['quality_score'][di]:.2f}",
                              f'A C {ac:.2f} Q {aq:.2f}',
                              f'P {person:.2f} coverage {coverage:.2f}' if person >= 0 else 'P: no usable ROI')
                track.observed = self.frame
            good = ac >= cfg.ambient_high and aq >= 0.2
            bad = ac < cfg.ambient_low or aq < 0.15
            track.ambient_good = track.ambient_good + 1 if good else 0
            track.ambient_bad = track.ambient_bad + 1 if bad else 0
            if track.ambient_good >= 3:
                track.need_roi = False
            if track.ambient_bad >= 2 or negative:
                track.need_roi = True
            if track.state[3] < cfg.roi_height - 10:
                track.far = True
            elif track.state[3] > cfg.roi_height + 10:
                track.far = False
            if track.negatives >= cfg.negative_hits:
                self.memory.remember(boxes[di] if di is not None else state_box(track.state), ambient, now)
            elif track.misses < cfg.max_misses:
                retained.append(track)
        self.tracks = retained

    def candidates(self):
        boxes = []
        for track in self.tracks:
            interval = self.args.stable_roi_interval if track.far and track.stable >= 3 else 1
            required = not track.confirmed or track.far or track.need_roi or track.misses > 0
            if track.uncertain:
                interval = max(interval, min(2 ** min(track.uncertain, 4), 15))
            if required and self.frame + 1 - track.last_roi >= interval:
                box = np.clip(state_box(track.state + track.velocity), 0, [640, 480, 640, 480])
                if (box[2:] > box[:2]).all():
                    boxes.append(box)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description='V4 candidate verification and alpha-beta tracking.')
    parser.add_argument('model_dir')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video')
    parser.add_argument('--log', help='Optional per-frame JSONL for reviewing false confirmations and ROI use.')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--track-confidence', type=float, default=0.10)
    parser.add_argument('--person-high', type=float, default=0.70)
    parser.add_argument('--person-low', type=float, default=0.20)
    parser.add_argument('--confirm-hits', type=int, default=2)
    parser.add_argument('--negative-hits', type=int, default=2)
    parser.add_argument('--ambient-high', type=float, default=0.30)
    parser.add_argument('--ambient-low', type=float, default=0.20)
    parser.add_argument('--memory-seconds', type=float, default=30)
    parser.add_argument('--memory-change', type=float, default=0.04)
    parser.add_argument('--max-misses', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--match-iou', type=float, default=0.2)
    parser.add_argument('--roi-height', type=float, default=130)
    parser.add_argument('--stable-roi-interval', type=int, default=5)
    parser.add_argument('--roi-scale', type=float, default=1.15)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    detector, tracker = QuadROIV4Onnx(args.model_dir, args.threads), Tracker(args)
    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    log = open(args.log, 'w', encoding='utf-8') if args.log else None
    rois = np.empty((0, 4), dtype=np.float32)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            start = time.monotonic()
            frame = vga_frame(frame)
            result = detector.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rois,
                                      min(args.confidence, args.track_confidence))
            now = tracker.frame / video_fps if args.video else start
            tracker.update(result, frame[::8, 2::8, 1], now)
            used = rois
            rois = next_rois(tracker.candidates(), detector.settings, args.roi_scale)
            if log:
                entry = dict(frame=tracker.frame, time=now, rois=used.tolist(),
                             detections={k: v.tolist() for k, v in result.items()},
                             tracks=[dict(id=t.id, box=state_box(t.state).tolist(),
                                          confirmed=t.confirmed, misses=t.misses) for t in tracker.tracks])
                log.write(json.dumps(entry) + '\n')
            for box in used.astype(int):
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 180, 0), 1)
            for track in tracker.tracks:
                box = np.clip(state_box(track.state), 0, [639, 479, 639, 479]).astype(int)
                status = 'coast' if track.misses else 'confirmed' if track.confirmed else 'pending'
                color = (0, 180, 255) if track.misses else (0, 220, 0) if track.confirmed else (0, 255, 255)
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                age = tracker.frame - track.observed
                lines = (f'#{track.id} {status} age={age}',) + track.text
                x, y = min(box[0], 350), max(40, min(box[1], 480 - len(lines) * 18))
                for line in lines:
                    width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
                    cv2.rectangle(frame, (x, y - 13), (x + width + 4, y + 3), (0, 0, 0), -1)
                    cv2.putText(frame, line, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                    y += 18
            fps = 1 / (time.monotonic() - start)
            cv2.putText(frame, f'{fps:.1f} processing FPS  ROIs {len(used)}/8  memory {len(tracker.memory.regions)}',
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Quad ROI v4', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if log:
            log.close()


if __name__ == '__main__':
    main()
