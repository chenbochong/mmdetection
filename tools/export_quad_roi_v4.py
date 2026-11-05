import argparse
import json
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from export_quad_roi_v3 import AmbientModel, FusionModel


class ROIModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder, self.head = model.backbone.roi_encoder, model.hd_head

    def forward(self, roi):
        hd, shallow = self.encoder(roi)
        return hd, self.head(hd), shallow


def main():
    parser = argparse.ArgumentParser(description='Export v4: ambient, ROI, fusion and per-candidate verifier.')
    parser.add_argument('checkpoint')
    parser.add_argument('--config', default='configs/ambient_train/quad_roi_detector_v4.py')
    parser.add_argument('--out-dir', default='work_dirs/quad_roi_v4_onnx')
    args = parser.parse_args()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.eval()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    graphs = [
        ('ambient', AmbientModel(model), (torch.zeros(1, 1, 60, 80),), ['ambient'], ['context', 'fine', 'coarse']),
        ('roi', ROIModel(model), (torch.zeros(1, 16, 24, 24),), ['roi'], ['hd', 'prediction', 'shallow']),
        ('fusion', FusionModel(model), (torch.zeros(1, 128, 12, 12), torch.zeros(1, 64, 12, 12),
                                      torch.ones(1, 1, 12, 12), torch.ones(1, 2, 12, 12)),
         ['hd', 'context', 'valid', 'scale'], ['prediction']),
        ('verifier', model.verifier, (torch.zeros(1, 32, 8, 8),), ['crop'], ['logit'])]
    with torch.no_grad():
        for name, graph, inputs, input_names, output_names in graphs:
            torch.onnx.export(graph.eval(), inputs, str(output / f'{name}.onnx'),
                              input_names=input_names, output_names=output_names, opset_version=13,
                              dynamo=False, do_constant_folding=True, external_data=False)
            print(output / f'{name}.onnx')
    roi_cfg = cfg.test_dataloader.dataset.pipeline[-1]
    metadata = dict(version=4, packed_size=24, checkpoint=str(Path(args.checkpoint).resolve()),
                    nms_iou=model.nms_iou, max_per_img=model.max_per_img,
                    ambient_topk=model.ambient_topk, roi_topk=model.roi_topk,
                    pre_verify_topk=model.pre_verify_topk, min_visibility=model.min_visibility,
                    support_iou=model.support_iou, merge_cfg=dict(roi_cfg.merge_cfg),
                    merge_scale_max=max(roi_cfg.roi_scale))
    (output / 'quad_roi.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
