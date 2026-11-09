import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from quad_roi_runtime import nms, pack_roi, sigmoid


def add_roi(canvas, feature, box, affine):
    origin = np.array([2.5, 0.5], dtype=np.float32)
    x1, y1 = np.maximum(0, np.ceil((box[:2] - origin) / 8).astype(int))
    x2, y2 = np.minimum([80, 60], np.ceil((box[2:] - origin) / 8).astype(int))
    c = feature[0]
    x = np.arange(x1, x2, dtype=np.float32) * 8 + origin[0]
    y = np.arange(y1, y2, dtype=np.float32) * 8 + origin[1]
    u = np.clip((x * affine[0] + affine[2]) * c.shape[2] - 0.25, 0, c.shape[2] - 1)
    v = np.clip((y * affine[1] + affine[3]) * c.shape[1] - 0.25, 0, c.shape[1] - 1)
    left, top = u.astype(int), v.astype(int)[:, None]
    right, bottom = np.minimum(left + 1, c.shape[2] - 1), np.minimum(top + 1, c.shape[1] - 1)
    wx, wy = (u - left).astype(np.float32), (v[:, None] - top).astype(np.float32)
    sampled = ((1 - wy) * ((1 - wx) * c[:, top, left] + wx * c[:, top, right]) +
               wy * ((1 - wx) * c[:, bottom, left] + wx * c[:, bottom, right]))
    canvas[0, :-1, y1:y2, x1:x2] += sampled
    canvas[0, -1, y1:y2, x1:x2] += 1


class QuadROIV5Onnx:
    def __init__(self, model_dir, threads=1):
        path = Path(model_dir)
        self.settings = json.loads((path / 'quad_roi.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.sessions = {name: ort.InferenceSession(str(path / f'{name}.onnx'), options,
                                                     providers=['CPUExecutionProvider'])
                         for name in ('ambient', 'roi', 'global')}

    def predict(self, rgb_vga, rois, confidence_threshold):
        rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
        image = np.ascontiguousarray(rgb_vga[::8, 2::8, 1][None, None], dtype=np.float32) / 255
        outputs = self.sessions['ambient'].run(None, {'image': image})
        canvas = np.zeros((1, 17, 60, 80), dtype=np.float32)
        for box in rois:
            packed, affine = pack_roi(rgb_vga, box, side=24)
            features = self.sessions['roi'].run(None, {'roi': packed})[0]
            add_roi(canvas, features, box, affine)
        count = canvas[:, -1:]
        canvas[:, :-1] /= np.maximum(count, 1)
        np.minimum(count, 1, out=count)
        inputs = dict(zip(('fine', 'medium', 'coarse'), outputs[:3]))
        predictions = self.sessions['global'].run(None, dict(inputs, canvas=canvas))
        cfg = self.settings
        candidates = []
        for pred, aux, stride, (low, high) in zip(predictions, outputs[3:], cfg['strides'], cfg['regress_ranges']):
            cls, sides, ctr = pred[0, 0], pred[0, 1:5], pred[0, 5]
            score = np.sqrt(sigmoid(cls) * sigmoid(ctr)).ravel()
            idx = np.flatnonzero(score >= confidence_threshold)
            idx = idx[np.argsort(-score[idx], kind='stable')[:cfg['pre_nms']]]
            y, x = np.unravel_index(idx, cls.shape)
            xy = (np.stack((x, y), axis=1).astype(np.float32) + 0.5) * stride
            sides = sides[:, y, x].T
            boxes = np.column_stack((xy - sides[:, :2], xy + sides[:, 2:]))
            boxes = np.clip(boxes, 0, [640, 480, 640, 480]).astype(np.float32)
            covered = ((xy[:, None] >= rois[None, :, :2]) & (xy[:, None] < rois[None, :, 2:])).all(-1).any(1)
            distance = sides.max(axis=1)
            support = covered & (distance >= low) & (distance <= high)
            candidates.append((boxes, score[idx], support, sigmoid(aux[0, 0, y, x]), xy,
                               np.full(len(idx), stride, dtype=np.int32)))
        boxes, scores, support, ambient, points, strides = map(np.concatenate, zip(*candidates))
        valid = np.flatnonzero((boxes[:, 2:] > boxes[:, :2]).all(1))
        keep = valid[nms(boxes[valid], scores[valid], cfg['nms_iou'], cfg['max_per_img'])]
        return dict(bboxes=boxes[keep], scores=scores[keep], hd_support=support[keep].astype(np.uint8),
                    ambient_score=ambient[keep], points=points[keep], strides=strides[keep])

    def detect(self, rgb_vga, rois, confidence_threshold):
        result = self.predict(rgb_vga, rois, confidence_threshold)
        return result['bboxes'], result['scores'], result['hd_support']
