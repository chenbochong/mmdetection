import numpy as np

from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackAmbient(BaseTransform):

    def transform(self, results: dict) -> dict:
        img = results['img']
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        H, W, C = img.shape

        bboxes = results['gt_bboxes'].cxcywh.cpu().numpy()
        labels = results['gt_bboxes_labels']
        ignore_flags = results['gt_ignore_flags']

        bboxes_to_merge = []
        for (cx, cy, w, h), label, ignore_flag in zip(bboxes, labels, ignore_flags):
            if ignore_flag:
                continue
            area = w * h
            if area > 350 * 480:
                if np.random.rand() > 0.2:
                    continue
            elif area > 120 * 240:
                if np.random.rand() > 0.4:
                    continue
            elif area > 70 * 140:
                if np.random.rand() > 0.6:
                    continue
            else:
                if np.random.rand() > 0.8:
                    continue
            cx = np.clip(cx + np.random.normal(0, 0.2) * w, 1, W - 1)
            cy = np.clip(cy + np.random.normal(0, 0.2) * h, 1, H - 1)
            x1 = np.clip(int(cx - w / 2), 0, W - 1)
            y1 = np.clip(int(cy - h / 2), 0, H - 1)
            x2 = np.clip(int(cx + w / 2), x1 + 1, W)
            y2 = np.clip(int(cy + h / 2), y1 + 1, H)
            bboxes_to_merge.append([x1, y1, x2, y2])
        for i in range(np.random.randint(0, 5)):
            random_draw = np.random.rand()
            if random_draw < 0.2:
                w = np.random.normal(120, 20)
                h = np.random.normal(240, 40)
            elif random_draw < 0.8:
                w = np.random.normal(70, 10)
                h = np.random.normal(140, 20)
            else:
                w = np.random.normal(35, 5)
                h = np.random.normal(70, 10)
            w = np.clip(int(w), 1, W)
            h = np.clip(int(h), 1, H)
            cx = np.random.randint(1 + w // 2, W - w // 2)
            cy = np.random.randint(1 + h // 2, H - h // 2)
            bboxes_to_merge.append([int(cx - w // 2), int(cy - h // 2), int(cx + w // 2), int(cy + h // 2)])

        merged_bboxes = self.group_bboxes(bboxes_to_merge)
        patch_cells = set()
        for x1, y1, x2, y2 in merged_bboxes:
            gx1 = int(np.floor(x1 / 32))
            gy1 = int(np.floor(y1 / 32))
            gx2 = int(np.ceil(x2 / 32))
            gy2 = int(np.ceil(y2 / 32))
            gx1 = np.clip(gx1, 0, W // 32 - 1)
            gy1 = np.clip(gy1, 0, H // 32 - 1)
            gx2 = np.clip(gx2, gx1 + 1, W // 32)
            gy2 = np.clip(gy2, gy1 + 1, H // 32)
            for gy in range(gy1, gy2):
                for gx in range(gx1, gx2):
                    patch_cells.add((gx, gy))

        patch_cells = sorted(patch_cells, key=lambda p: (p[1], p[0]))
        roi_patches = []
        roi_grid_xy = []

        for gx, gy in patch_cells:
            patch = self.pack_quad_bayer_patch(img, gx * 32, gy * 32)
            roi_patches.append(patch)
            roi_grid_xy.append((gx, gy))

        if len(roi_patches) > 0:
            roi_patches = np.stack(roi_patches, axis=0).astype(np.float32)
            roi_grid_xy = np.asarray(roi_grid_xy, dtype=np.int64)
        else:
            roi_patches = np.empty((0, 8, 8, 16), dtype=np.float32)
            roi_grid_xy = np.empty((0, 2), dtype=np.int64)

        results['img'] = img[::8, 2::8, 1]
        results['roi_img'] = roi_patches
        results['roi_grid_xy'] = roi_grid_xy
        return results

    def pack_quad_bayer_patch(self, img, x, y):
        block = img[y:y + 32, x:x + 32]

        patch = np.stack([
            # R / channel 2, 左上 2x2
            block[0::4, 0::4, 2],
            block[0::4, 1::4, 2],
            block[1::4, 0::4, 2],
            block[1::4, 1::4, 2],
            # G / channel 1, 右上 2x2
            block[0::4, 2::4, 1],
            block[0::4, 3::4, 1],
            block[1::4, 2::4, 1],
            block[1::4, 3::4, 1],
            # G / channel 1, 左下 2x2
            block[2::4, 0::4, 1],
            block[2::4, 1::4, 1],
            block[3::4, 0::4, 1],
            block[3::4, 1::4, 1],
            # B / channel 0, 右下 2x2
            block[2::4, 2::4, 0],
            block[2::4, 3::4, 0],
            block[3::4, 2::4, 0],
            block[3::4, 3::4, 0],
        ], axis=-1)
        return patch.astype(np.float32, copy=False)

    def area(self, box):
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def merge(self, box1, box2):
        return np.array([
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ], dtype=np.float32)

    def group_bboxes(self, bboxes, max_boxes=7, extra_cost_allowance=3000):
        boxes = np.asarray(bboxes, dtype=np.float32)
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        while len(boxes) > 1:
            best_i = None
            best_j = None
            best_box = None
            best_cost = float("inf")

            # Find the cheapest pair to merge
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    merged = self.merge(boxes[i], boxes[j])
                    old_area = self.area(boxes[i]) + self.area(boxes[j])
                    new_area = self.area(merged)
                    cost = new_area - old_area
                    if cost < best_cost:
                        best_cost = cost
                        best_i = i
                        best_j = j
                        best_box = merged

            # We MUST merge if we're still above the box limit.
            must_merge = len(boxes) > max_boxes
            # Once <= max_boxes, only continue if merging is sufficiently cheap.
            cheap_extra_merge = best_cost <= extra_cost_allowance
            if not must_merge and not cheap_extra_merge:
                break

            # Remove the two old boxes and insert the merged one
            new_boxes = []
            for k in range(len(boxes)):
                if k != best_i and k != best_j:
                    new_boxes.append(boxes[k])
            new_boxes.append(best_box)
            boxes = np.asarray(new_boxes, dtype=np.float32)

        return boxes

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str