import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Browse mined negative boxes and person annotations.')
    parser.add_argument('annotations')
    parser.add_argument('image_dir')
    parser.add_argument('--save-dir', default='work_dirs/mined_preview')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    data = json.loads(Path(args.annotations).read_text(encoding='utf-8'))
    images = [item for item in data['images'] if item.get('hard_negative_boxes')]
    if args.shuffle:
        random.Random(2026).shuffle(images)
    if not images:
        print('No hard_negative_boxes found.')
        return
    person_ids = {item['id'] for item in data['categories'] if item['name'] == 'person'}
    people = {}
    for ann in data['annotations']:
        if ann['category_id'] in person_ids:
            people.setdefault(ann['image_id'], []).append(ann)
    index = args.start % len(images)
    cv2.namedWindow('Mined negatives', cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            info = images[index]
            path = Path(args.image_dir) / info['file_name']
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            scale = min(1200 / w, 800 / h, 2)
            nw, nh = round(w * scale), round(h * scale)
            canvas = cv2.copyMakeBorder(cv2.resize(image, (nw, nh)), 70, 0, 0, 0,
                                        cv2.BORDER_CONSTANT)

            def draw(box, color, label):
                coords = np.asarray(box) * [nw / w, nh / h, nw / w, nh / h]
                x1, y1, x2, y2 = np.rint(coords).astype(int)
                y1, y2 = y1 + 70, y2 + 70
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, label, (x1, max(83, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            for ann in people.get(info['id'], []):
                x, y, bw, bh = ann['bbox']
                ignored = ann.get('iscrowd', 0) or ann.get('ignore', 0)
                draw([x, y, x + bw, y + bh], (255, 255, 0) if ignored else (0, 220, 0),
                     'person ignore' if ignored else 'person GT')
            for number, box in enumerate(info['hard_negative_boxes'], 1):
                draw(box, (0, 0, 255), f'NEG {number}')
            lines = [f'{index + 1}/{len(images)}  image_id={info["id"]}  negatives={len(info["hard_negative_boxes"])}',
                     'RED: negative  GREEN: person  CYAN: ignored person',
                     'D/Space: next   A: previous   S: save   Q/Esc: quit']
            for row, line in enumerate(lines):
                cv2.putText(canvas, line, (8, 18 + row * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1, cv2.LINE_AA)
            print(f'[{index + 1}/{len(images)}] {path}')
            cv2.imshow('Mined negatives', canvas)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                directory = Path(args.save_dir)
                directory.mkdir(parents=True, exist_ok=True)
                output = directory / f'{info["id"]}_negatives.png'
                cv2.imencode('.png', canvas)[1].tofile(output)
                print(f'Saved {output}')
            elif key == ord('a'):
                index = (index - 1) % len(images)
            elif key in (ord('d'), ord(' ')):
                index = (index + 1) % len(images)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
