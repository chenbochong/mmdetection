import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


class Ambient(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.ambient

    def forward(self, image):
        levels, scores = self.encoder(image)
        return (*levels, *scores)


class Global(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fusion, self.head = model.fusion, model.head

    def forward(self, fine, medium, coarse, canvas):
        outputs = self.head(self.fusion((fine, medium, coarse), canvas))
        return tuple(torch.cat(level, dim=1) for level in zip(*outputs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/ambient_train/quad_roi_detector_v5.py')
    parser.add_argument('--out-dir', default='work_dirs/quad_roi_v5_onnx')
    args = parser.parse_args()
    register_all_modules()
    config = Config.fromfile(args.config)
    model = MODELS.build(config.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.eval()
    path = Path(args.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    levels = ['fine', 'medium', 'coarse']
    features = tuple(torch.zeros(1, 32, 480 // stride, 640 // stride) for stride in model.head.strides)
    graphs = [
        ('ambient', Ambient(model), (torch.zeros(1, 1, 60, 80),), ['image'],
         levels + ['presence_' + name for name in levels]),
        ('roi', model.roi, (torch.zeros(1, 16, 24, 24),), ['roi'], ['features']),
        ('global', Global(model), (*features, torch.zeros(1, 17, 60, 80)),
         levels + ['canvas'], ['pred_' + name for name in levels])]
    with torch.no_grad():
        for name, graph, inputs, input_names, output_names in graphs:
            torch.onnx.export(graph.eval(), inputs, str(path / f'{name}.onnx'),
                              input_names=input_names, output_names=output_names,
                              opset_version=13, dynamo=False, external_data=False)
    crop = config.test_dataloader.dataset.pipeline[-1]
    settings = dict(version=5, strides=model.head.strides, regress_ranges=model.head.regress_ranges,
                    pre_nms=model.pre_nms, nms_iou=model.nms_iou, max_per_img=model.max_per_img,
                    merge_cfg=dict(crop.merge_cfg), merge_scale_max=max(crop.roi_scale))
    (path / 'quad_roi.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
