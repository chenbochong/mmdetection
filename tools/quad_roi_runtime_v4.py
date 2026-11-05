import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from quad_roi_runtime import align_context, box_area, box_iou, decode, nms, pack_roi, sigmoid


def verifier_crop(feature, box):
    t = (np.arange(8, dtype=np.float32) + 0.5) / 8
    x, y = np.meshgrid(t, t)
    xy = box[:2] + np.stack((x, y), -1) * (box[2:] - box[:2])
    c = feature[0]
    grid = xy * np.array([c.shape[2], c.shape[1]], dtype=np.float32) - 0.5
    gx, gy = np.clip(grid[..., 0], 0, c.shape[2] - 1), np.clip(grid[..., 1], 0, c.shape[1] - 1)
    x0, y0 = gx.astype(int), gy.astype(int)
    x1, y1 = np.minimum(x0 + 1, c.shape[2] - 1), np.minimum(y0 + 1, c.shape[1] - 1)
    wx, wy = (gx - x0).astype(np.float32), (gy - y0).astype(np.float32)
    value = ((1 - wy) * ((1 - wx) * c[:, y0, x0] + wx * c[:, y0, x1]) +
             wy * ((1 - wx) * c[:, y1, x0] + wx * c[:, y1, x1]))
    return np.ascontiguousarray(value[None])


class QuadROIV4Onnx:
    def __init__(self, model_dir, threads=1):
        directory = Path(model_dir)
        self.settings = json.loads((directory / 'quad_roi.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        for name in ('ambient', 'roi', 'fusion', 'verifier'):
            setattr(self, name, ort.InferenceSession(str(directory / f'{name}.onnx'), options,
                                                    providers=['CPUExecutionProvider']))

    def detect(self, rgb_vga, rois, confidence_threshold):
        result = self.predict(rgb_vga, rois, confidence_threshold)
        keep = result['scores'] >= confidence_threshold
        return result['bboxes'][keep], result['scores'][keep]

    def predict(self, rgb_vga, rois, confidence_threshold):
        cfg = self.settings
        rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
        ambient = np.ascontiguousarray(rgb_vga[::8, 2::8, 1][None, None], dtype=np.float32) / 255
        context, fine, coarse = self.ambient.run(None, {'ambient': ambient})
        boxes, centers, qualities, sources = [], [], [], []
        for pred, rows in ((fine, 30), (coarse, 15)):
            valid = np.zeros(pred.shape[-2:], dtype=np.float32)
            valid[:rows] = 1
            b, rank, parts = decode(pred, valid, cfg['ambient_topk'])
            boxes.append(b * np.array([640, 512, 640, 512], dtype=np.float32))
            centers.append(parts[:, 0] * (rank > 0))
            qualities.append(parts[:, 1])
            sources.append(np.zeros(len(b), dtype=np.int64))
        ambient_boxes, ambient_centers, ambient_quality = np.concatenate(boxes), np.concatenate(centers), np.concatenate(qualities)
        affines, shallow = [], []
        for roi in rois:
            packed, affine = pack_roi(rgb_vga, roi, side=24)
            hd, _, early = self.roi.run(None, {'roi': packed})
            aligned, valid = align_context(context, roi, affine, size=12)
            dimensions = (roi[2:] - roi[:2]) / np.array([640, 480], dtype=np.float32)
            scale = np.broadcast_to(dimensions[None, :, None, None], (1, 2, 12, 12)).copy()
            pred = self.fusion.run(None, dict(hd=hd, context=aligned, valid=valid, scale=scale))[0]
            b, rank, parts = decode(pred, valid[0, 0], cfg['roi_topk'], ltrb=True)
            boxes.append((b - np.tile(affine[2:], 2)) / np.tile(affine[:2], 2))
            centers.append(parts[:, 0] * (rank > 0))
            qualities.append(parts[:, 1])
            sources.append(np.ones(len(b), dtype=np.int64))
            affines.append(affine)
            shallow.append(early)
        boxes = np.clip(np.concatenate(boxes), 0, [640, 480, 640, 480]).astype(np.float32)
        center, quality, source = np.concatenate(centers), np.concatenate(qualities), np.concatenate(sources)
        keep = np.flatnonzero((center >= confidence_threshold) & (boxes[:, 2:] > boxes[:, :2]).all(1))
        keep = keep[np.argsort(-(center[keep] * quality[keep]), kind='stable')[:cfg['pre_verify_topk']]]
        boxes, center, quality, source = boxes[keep], center[keep], quality[keep], source[keep]
        count = len(boxes)
        result = dict(bboxes=boxes, center_score=center, quality_score=quality, source=source,
                      person_score=np.full(count, -1, dtype=np.float32),
                      verify_coverage=np.zeros(count, dtype=np.float32),
                      roi_id=np.full(count, -1, dtype=np.int64),
                      ambient_center=np.zeros(count, dtype=np.float32),
                      ambient_quality=np.zeros(count, dtype=np.float32))
        if count:
            matched = (box_iou(boxes, ambient_boxes) >= cfg['support_iou']) * ambient_centers[None]
            best = matched.argmax(1)
            result['ambient_center'] = matched[np.arange(count), best]
            result['ambient_quality'] = ambient_quality[best] * (result['ambient_center'] > 0)
        if count and len(rois):
            affines = np.asarray(affines)
            lo, hi = np.maximum(boxes[:, None, :2], rois[None, :, :2]), np.minimum(boxes[:, None, 2:], rois[None, :, 2:])
            visible = np.maximum(hi - lo, 0)
            coverage = visible.prod(-1) / np.maximum(box_area(boxes)[:, None], 1)
            pixels = visible * affines[None, :, :2] * 96
            valid = (coverage >= cfg['min_visibility']) & (pixels >= [4, 12]).all(-1)
            slots = (coverage * valid).argmax(1)
            for i, slot in enumerate(slots):
                if not valid[i, slot]:
                    continue
                window = np.r_[lo[i, slot], hi[i, slot]]
                affine = affines[slot]
                window = window * np.tile(affine[:2], 2) + np.tile(affine[2:], 2)
                logit = self.verifier.run(None, {'crop': verifier_crop(shallow[slot], window)})[0]
                result['person_score'][i] = sigmoid(logit).item()
                result['verify_coverage'][i] = coverage[i, slot]
                result['roi_id'][i] = slot
        person = result['person_score']
        result['scores'] = center * np.where(person >= 0, person, 1)
        keep = nms(boxes, result['scores'] * quality, cfg['nms_iou'], cfg['max_per_img'])
        return {name: value[keep] for name, value in result.items()}
