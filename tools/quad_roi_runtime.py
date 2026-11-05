import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def intersection_area(a, b):
    size = np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    return np.maximum(size, 0).prod(axis=-1)


def box_area(boxes):
    return np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=-1)


def box_iou(a, b):
    intersection = intersection_area(a, b)
    return intersection / np.maximum(box_area(a)[:, None] + box_area(b)[None] - intersection, 1e-6)


def nms(boxes, scores, threshold, limit):
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order) and len(keep) < limit:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[box_iou(boxes[i:i + 1], boxes[rest])[0] <= threshold]
    return np.asarray(keep, dtype=np.int64)


def pack_roi(rgb, box, side=16):
    x1, y1, x2, y2 = box.astype(int)
    block = rgb[y1:y2, x1:x2]
    # RGB input; the 16 channels retain the row-major Quad Bayer phases.
    phases = [block[y::4, x::4, 0 if y < 2 and x < 2 else
                    2 if y >= 2 and x >= 2 else 1]
              for y in range(4) for x in range(4)]
    packed = np.stack(phases, axis=-1).astype(np.float32)
    ph, pw = packed.shape[:2]
    ratio = side / max(ph, pw)
    nw, nh = max(1, round(pw * ratio)), max(1, round(ph * ratio))
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    resized = np.stack([cv2.resize(packed[..., c], (nw, nh), interpolation=interpolation)
                        for c in range(16)], axis=-1)
    px, py = (side - nw) // 2, (side - nh) // 2
    canvas = np.zeros((side, side, 16), dtype=np.float32)
    canvas[py:py + nh, px:px + nw] = resized
    scale = np.array([nw / (side * (x2 - x1)), nh / (side * (y2 - y1))])
    offset = np.array([px / side, py / side]) - box[:2] * scale
    affine = np.r_[scale, offset].astype(np.float32)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None] / 255), affine


def align_context(feature, box, affine, size=8):
    t = (np.arange(size, dtype=np.float32) + 0.25) / size
    x, y = np.meshgrid(t, t)
    world = (np.stack((x, y), axis=-1) - affine[2:]) / affine[:2]
    valid = ((world >= box[:2]) & (world < box[2:])).all(axis=-1).astype(np.float32)
    grid = (world - np.array([2.5, 0.5], dtype=np.float32)) / 16
    c = feature[0]
    gx = np.clip(grid[..., 0], 0, c.shape[2] - 1)
    gy = np.clip(grid[..., 1], 0, c.shape[1] - 1)
    x0, y0 = gx.astype(int), gy.astype(int)
    x1, y1 = np.minimum(x0 + 1, c.shape[2] - 1), np.minimum(y0 + 1, c.shape[1] - 1)
    wx, wy = (gx - x0).astype(np.float32), (gy - y0).astype(np.float32)
    aligned = ((1 - wy) * ((1 - wx) * c[:, y0, x0] + wx * c[:, y0, x1]) +
               wy * ((1 - wx) * c[:, y1, x0] + wx * c[:, y1, x1]))
    return np.ascontiguousarray((aligned * valid)[None]), valid[None, None]


def sigmoid(x):
    return (np.tanh(x * 0.5) + 1) * 0.5


def decode(prediction, valid, topk, ltrb=False):
    pred = prediction[0]
    h, w = pred.shape[-2:]
    offset_index = 5 if ltrb else 3
    quality_index = offset_index + 2
    heat = sigmoid(pred[0]) * valid
    peak = heat == cv2.dilate(heat, np.ones((3, 3), dtype=np.uint8))
    scores = (heat * sigmoid(pred[quality_index]) * peak).ravel()
    indices = np.argsort(-scores, kind='stable')[:topk]
    selected = pred.reshape(pred.shape[0], -1)[:, indices]
    grid = np.stack((indices % w, indices // w)).astype(np.float32)
    point = grid + sigmoid(selected[offset_index:quality_index])
    size = np.maximum(selected[1:offset_index], 0) + 1e-3
    if ltrb:
        boxes = np.concatenate((point - size[:2], point + size[2:]), axis=0)
    else:
        boxes = np.concatenate((point - size / 2, point + size / 2), axis=0)
    boxes = boxes.T / np.array([w, h, w, h], dtype=np.float32)
    components = np.stack((sigmoid(selected[0]), sigmoid(selected[quality_index])), axis=1)
    return boxes, scores[indices], components


class QuadROIOnnx:
    def __init__(self, model_dir, threads=1):
        directory = Path(model_dir)
        self.settings = json.loads((directory / 'quad_roi.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.ambient = ort.InferenceSession(str(directory / 'ambient.onnx'), options,
                                           providers=['CPUExecutionProvider'])
        self.roi = ort.InferenceSession(str(directory / 'roi.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        self.fusion = ort.InferenceSession(str(directory / 'fusion.onnx'), options,
                                          providers=['CPUExecutionProvider'])

    def detect(self, rgb_vga, rois, confidence_threshold):
        """RGB uint8 HWC 480x640; supplied xyxy ROIs are valid, 4-aligned, and at most eight."""
        result = self.predict(rgb_vga, rois, confidence_threshold)
        return result['bboxes'], result['scores']

    def predict(self, rgb_vga, rois, confidence_threshold):
        settings = self.settings
        rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
        ambient = np.ascontiguousarray(rgb_vga[::8, 2::8, 1][None, None], dtype=np.float32) / 255
        context, fine, coarse = self.ambient.run(['context', 'fine', 'coarse'], {'ambient': ambient})
        ambient_boxes, ambient_scores, ambient_components = [], [], []
        for pred, rows in [(fine, 30), (coarse, 15)]:
            valid = np.zeros(pred.shape[-2:], dtype=np.float32)
            valid[:rows] = 1
            boxes, scores, components = decode(pred, valid, settings['ambient_topk'])
            ambient_components.append(components)
            ambient_boxes.append(boxes * np.array([640, 512, 640, 512], dtype=np.float32))
            ambient_scores.append(scores)
        ambient_boxes, ambient_scores = np.concatenate(ambient_boxes), np.concatenate(ambient_scores)
        ambient_components = np.concatenate(ambient_components)
        candidate_components = [ambient_components]
        hd_components = []
        candidates, confidences = [ambient_boxes], [ambient_scores]
        sources = [np.zeros(len(ambient_scores), dtype=np.int64)]
        hd_boxes, hd_scores, hd_slots = [], [], []
        for slot, box in enumerate(rois):
            packed, affine = pack_roi(rgb_vga, box)
            hd, hd_pred = self.roi.run(['hd', 'prediction'], {'roi': packed})
            aligned, valid = align_context(context, box, affine)
            dimensions = (box[2:] - box[:2]) / np.array([640, 480], dtype=np.float32)
            scale = np.broadcast_to(dimensions[None, :, None, None], (1, 2, 8, 8)).copy()
            fused_pred = self.fusion.run(['prediction'], dict(
                hd=hd, context=aligned, valid=valid, scale=scale))[0]
            for pred, ltrb in [(fused_pred, True), (hd_pred, False)]:
                boxes, scores, components = decode(pred, valid[0, 0], settings['roi_topk'], ltrb)
                boxes = (boxes - np.tile(affine[2:], 2)) / np.tile(affine[:2], 2)
                if ltrb:
                    candidate_components.append(components)
                    candidates.append(boxes)
                    confidences.append(scores)
                    sources.append(np.ones(len(scores), dtype=np.int64))
                else:
                    hd_components.append(components)
                    hd_boxes.append(boxes)
                    hd_scores.append(scores)
                    hd_slots.append(np.full(len(scores), slot, dtype=np.int64))
        boxes, scores, sources = np.concatenate(candidates), np.concatenate(confidences), np.concatenate(sources)
        components = np.concatenate(candidate_components)
        boxes = np.clip(boxes, 0, np.array([640, 480, 640, 480], dtype=np.float32))
        keep = (scores >= confidence_threshold) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        boxes, scores, sources = boxes[keep], scores[keep], sources[keep]
        components = components[keep]
        keep = nms(boxes, scores, settings['nms_iou'], settings['max_per_img'])
        boxes, scores, sources = boxes[keep], scores[keep], sources[keep]
        components = components[keep]
        count = len(boxes)
        result = dict(bboxes=boxes, scores=scores, source=sources,
                      center_score=components[:, 0], quality_score=components[:, 1],
                      ambient_center=np.zeros(count, dtype=np.float32),
                      ambient_quality=np.zeros(count, dtype=np.float32),
                      hd_center=np.zeros(count, dtype=np.float32),
                      hd_quality=np.zeros(count, dtype=np.float32),
                      ambient_score=np.zeros(count, dtype=np.float32),
                      hd_support=np.zeros(count, dtype=np.float32),
                      hd_coverage=np.zeros(count, dtype=np.float32),
                      hd_available=np.zeros(count, dtype=bool),
                      roi_id=np.full(count, -1, dtype=np.int64))
        if not count:
            return result
        overlap = box_iou(boxes, ambient_boxes)
        result['ambient_score'] = (ambient_scores[None] * (overlap >= settings['support_iou'])).max(axis=1)
        ambient_match = (ambient_scores[None] * (overlap >= settings['support_iou'])).argmax(axis=1)
        has_ambient = result['ambient_score'] > 0
        result['ambient_center'][has_ambient] = ambient_components[ambient_match[has_ambient], 0]
        result['ambient_quality'][has_ambient] = ambient_components[ambient_match[has_ambient], 1]
        if len(rois):
            coverage = intersection_area(boxes, rois) / np.maximum(box_area(boxes)[:, None], 1)
            best_roi = coverage.argmax(axis=1)
            result['hd_coverage'] = coverage[np.arange(count), best_roi]
            result['hd_available'] = result['hd_coverage'] > 0
            result['roi_id'] = np.where(result['hd_available'], best_roi, -1)
            evidence, evidence_scores, slots = np.concatenate(hd_boxes), np.concatenate(hd_scores), np.concatenate(hd_slots)
            windows = rois[slots]
            visible = np.concatenate((np.maximum(boxes[:, None, :2], windows[None, :, :2]),
                                      np.minimum(boxes[:, None, 2:], windows[None, :, 2:])), axis=2)
            visible_area = np.maximum(visible[..., 2:] - visible[..., :2], 0).prod(axis=2)
            intersection = np.maximum(np.minimum(visible[..., 2:], evidence[None, :, 2:]) -
                                      np.maximum(visible[..., :2], evidence[None, :, :2]), 0).prod(axis=2)
            iou = intersection / np.maximum(visible_area + box_area(evidence)[None] - intersection, 1)
            iof = intersection / np.maximum(box_area(evidence)[None], 1)
            matches = ((iou >= settings['support_iou']) & (iof >= settings['support_iof']) &
                       (coverage[:, slots] >= settings['min_visibility']))
            support = evidence_scores[None] * matches
            match = support.argmax(axis=1)
            result['hd_support'] = support[np.arange(count), match]
            supported = result['hd_support'] > 0
            evidence_components = np.concatenate(hd_components)
            result['hd_center'][supported] = evidence_components[match[supported], 0]
            result['hd_quality'][supported] = evidence_components[match[supported], 1]
            result['roi_id'][supported] = slots[match[supported]]
            result['hd_coverage'][supported] = coverage[np.arange(count)[supported], slots[match[supported]]]
        return result
