import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


class AmbientModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.fine_head = model.ambient_head
        self.coarse_head = model.coarse_head

    def forward(self, ambient):
        p2, p3 = self.backbone.ambient(ambient)
        return p2, self.fine_head(p2), self.coarse_head(p3)


class ROIModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.backbone.roi_encoder
        self.head = model.hd_head

    def forward(self, roi):
        hd = self.encoder(roi)
        return hd, self.head(hd)


class FusionModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.neck = model.neck
        self.head = model.fused_head

    def forward(self, hd, context, valid, scale):
        context = self.neck.context_projection(context) * valid
        return self.head(self.neck.layers(torch.cat((hd, context, valid, scale), dim=1)))


def main():
    parser = argparse.ArgumentParser(description='Export the three fixed-shape v3 CNNs.')
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/ambient_train/quad_roi_detector_v3.py')
    parser.add_argument('--out-dir', default='work_dirs/quad_roi_v3_onnx')
    args = parser.parse_args()

    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.eval()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    ac = cfg.model.backbone.ambient_channels
    rc = cfg.model.backbone.roi_channels
    graphs = [
        ('ambient', AmbientModel(model), (torch.zeros(1, 1, 60, 80),),
         ['ambient'], ['context', 'fine', 'coarse']),
        ('roi', ROIModel(model), (torch.zeros(1, 16, 16, 16),),
         ['roi'], ['hd', 'prediction']),
        ('fusion', FusionModel(model),
         (torch.zeros(1, rc, 8, 8), torch.zeros(1, ac, 8, 8),
          torch.ones(1, 1, 8, 8), torch.ones(1, 2, 8, 8)),
         ['hd', 'context', 'valid', 'scale'], ['prediction'])]
    with torch.no_grad():
        for name, graph, inputs, input_names, output_names in graphs:
            torch.onnx.export(
                graph.eval(), inputs, str(output / f'{name}.onnx'),
                input_names=input_names, output_names=output_names,
                opset_version=13, dynamo=False, export_params=True,
                do_constant_folding=True, external_data=False)
            print(output / f'{name}.onnx')

    roi_cfg = cfg.test_dataloader.dataset.pipeline[-1]
    metadata = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        config=str(Path(args.config).resolve()),
        nms_iou=model.nms_iou, max_per_img=model.max_per_img,
        ambient_topk=model.ambient_topk, roi_topk=model.roi_topk,
        support_iou=model.support_iou, support_iof=model.support_iof,
        min_visibility=model.min_visibility,
        merge_cfg=dict(roi_cfg.merge_cfg), merge_scale_max=max(roi_cfg.roi_scale))
    (output / 'quad_roi.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
