from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class QuadROICocoDataset(CocoDataset):
    def __init__(self, negative_classes=('chair', 'couch', 'potted plant', 'vase',
                                        'backpack', 'suitcase', 'teddy bear'), **kwargs):
        self.negative_classes = set(negative_classes)
        super().__init__(**kwargs)

    def parse_data_info(self, raw_data_info):
        data = super().parse_data_info(raw_data_info)
        boxes = list(raw_data_info['raw_img_info'].get('hard_negative_boxes', []))
        for ann in raw_data_info['raw_ann_info']:
            if ann.get('ignore', False) or ann.get('iscrowd', False):
                continue
            if ann['category_id'] in self.cat_ids:
                continue
            if self.coco.cats[ann['category_id']]['name'] not in self.negative_classes:
                continue
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
        data['hard_negative_boxes'] = boxes
        return data
