import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from quad_roi_runtime import QuadROIOnnx, intersection_area, nms
from quad_roi_runtime_v4 import QuadROIV4Onnx


def main():
    parser = argparse.ArgumentParser(description='Mine training-only false windows into a new COCO JSON.')
    parser.add_argument('model_dir')
    parser.add_argument('annotations')
    parser.add_argument('image_dir')
    parser.add_argument('output')
    parser.add_argument('--confidence', type=float, default=0.15)
    parser.add_argument('--max-images', type=int, default=5000)
    args = parser.parse_args()
    metadata = json.loads((Path(args.model_dir) / 'quad_roi.json').read_text(encoding='utf-8'))
    detector = (QuadROIV4Onnx if metadata.get('version') == 4 else QuadROIOnnx)(args.model_dir)
    data = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    person_ids = {c['id'] for c in data['categories'] if c['name'] == 'person'}
    annotations = {}
    for ann in data['annotations']:
        if ann['category_id'] in person_ids:
            x, y, w, h = ann['bbox']
            annotations.setdefault(ann['image_id'], []).append([x, y, x + w, y + h])
    order = np.random.RandomState(2026).permutation(len(data['images']))[:args.max_images]
    total = 0
    for step, index in enumerate(order):
        info = data['images'][index]
        image = cv2.imread(str(Path(args.image_dir) / info['file_name']))
        h, w = image.shape[:2]
        ratio = min(640 / w, 480 / h)
        nw, nh = round(w * ratio), round(h * ratio)
        x, y = (640 - nw) // 2, (480 - nh) // 2
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[y:y + nh, x:x + nw] = cv2.cvtColor(cv2.resize(image, (nw, nh)), cv2.COLOR_BGR2RGB)
        scale, offset = np.array([nw / w, nh / h]), np.array([x, y])
        occupied = np.asarray(annotations.get(info['id'], []), dtype=np.float32).reshape(-1, 4)
        occupied = occupied * np.tile(scale, 2) + np.tile(offset, 2)
        first = detector.predict(rgb, [], args.confidence)
        proposals = first['bboxes'][:8]
        center, size = (proposals[:, :2] + proposals[:, 2:]) / 2, (proposals[:, 2:] - proposals[:, :2]) * 1.15
        lo = np.clip(np.floor((center - size / 2) / 4) * 4, 0, [636, 476])
        hi = np.clip(np.ceil((center + size / 2) / 4) * 4, lo + 4, [640, 480])
        second = detector.predict(rgb, np.concatenate((lo, hi), 1), args.confidence)
        boxes = np.concatenate((first['bboxes'], second['bboxes']))
        scores = np.concatenate((first['scores'], second['scores']))
        clear = ~(intersection_area(boxes, occupied) > 0).any(1)
        boxes, scores = boxes[clear], scores[clear]
        selected = nms(boxes, scores, 0.5, 8)
        boxes = (boxes[selected] - np.tile(offset, 2)) / np.tile(scale, 2)
        boxes = np.clip(boxes, 0, [w, h, w, h])
        boxes = boxes[(boxes[:, 2:] > boxes[:, :2]).all(1)]
        info.setdefault('hard_negative_boxes', []).extend(boxes.tolist())
        total += len(boxes)
        if (step + 1) % 100 == 0:
            print(f'{step + 1}/{len(order)} images, {total} mined windows')
    Path(args.output).write_text(json.dumps(data), encoding='utf-8')
    print(f'Saved {total} windows to {args.output}; review samples for missing person annotations.')


if __name__ == '__main__':
    main()
