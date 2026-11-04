# Quad ROI v2 fine-tuning

Sync `mmdet/models/dense_heads/quad_roi_head.py` and
`configs/ambient_train/quad_roi_detector_v2.py` to the training computer.
The v2 config inherits `quad_roi_detector.py`; keep that base config present.

V2 retains all network shapes, weights, inference operators and source outputs.
It changes only training supervision, the fine-tuning schedule and evaluation
settings. The original config keeps the original loss because
`direct_regression` defaults to False. No acceleration changes are included.

## Changes

- All three heads keep `2 * (1 - IoU)` and replace normalized-corner L1 with
  `1.0 * offset_L1 + 0.1 * width_height_L1`, evaluated at positive centers.
  Offsets are fractions within a feature cell; sizes are in feature-grid units.
  The terms remain grouped under `loss_box`. Its absolute value is not comparable
  with v1's `loss_box`. Heatmap and quality supervision are unchanged.
- Load learned weights and start a fresh AdamW optimizer and 15-epoch cosine
  schedule, from 2e-4 toward 1e-5, without the old 120-epoch schedule or warmup.
  Epoch numbering restarts at 1; weights do not restart from random initialization.
- Validation/test use at most three pseudo ROIs and score threshold 0.001.
  Training keeps the original ROI distribution (up to eight), including jitter,
  background ROIs and ROI dropout. Validation is not exactly three ROIs per frame.
- The 0.001 score threshold is for offline evaluation, not a recommended deployed
  decision threshold. Top-k limits, NMS and output caps are unchanged.

## Start training

Run from the repository root on the training computer:

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v2.py
```

The default starting checkpoint is
`work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth`, the best checkpoint
in the supplied log. If it has been replaced by a newer best checkpoint, change
`load_from` in the v2 config or override it:

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v2.py --cfg-options load_from=work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_40.pth
```

The epoch-40 filename is only an example; use a file that exists. Do not pass
`--resume` when first switching from v1 to v2. To resume an interrupted **v2** run:

```powershell
python tools/train.py configs/ambient_train/quad_roi_detector_v2.py --resume
```

Outputs go to `work_dirs/quad_roi_detector_v2`, separate from v1. Keep the chosen
v1 checkpoint for baseline comparisons. Check that the printed v2 configuration
shows 15 epochs, lr 0.0002, and the intended checkpoint before leaving it running.

## Same-checkpoint evaluations

These commands use the existing v1 checkpoint and require no training. Substitute
the actual checkpoint filename if necessary. All pseudo-ROI variants use the same
per-image seed and sampling parameters; changing the budget truncates the sorted
ROI list. The HD head's new loss option does not affect inference on old weights.

Three ROI budget, threshold 0.001 (v2's default validation setting):

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi3
```

Ambient only, same threshold:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_ambient --cfg-options test_dataloader.dataset.pipeline.2.pseudo_rois=False
```

Eight ROI budget, threshold 0.001, to isolate the threshold change against the
original eight-ROI/0.05 log:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi8_low --cfg-options test_dataloader.dataset.pipeline.2.max_rois=8
```

If the checkpoint or original validation settings differ from that log, also run
the same checkpoint at eight ROIs and threshold 0.05:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi8_original --cfg-options test_dataloader.dataset.pipeline.2.max_rois=8 model.score_thr=0.05
```

To evaluate a trained v2 model, substitute its checkpoint path in the same
commands. Compare v1 and v2 using identical ROI budgets and score thresholds;
the old 0.229 AP and default v2 AP use different evaluation settings. Changing
only the top-level `roi_settings` through CLI does not rebuild an already-defined
pipeline; the overrides above target the actual test pipeline transform.

No training, inference, or tests were run while implementing this change.
