# Quad ROI v2 fine-tuning

Sync `mmdet/datasets/transforms/quad_roi_crop.py`,
`mmdet/models/dense_heads/quad_roi_head.py`,
`mmdet/evaluation/metrics/quad_roi_coco_metric.py`,
`mmdet/evaluation/metrics/__init__.py` and
`configs/ambient_train/quad_roi_detector_v2.py` to the training computer.
The v2 config inherits `quad_roi_detector.py`; keep that base config present.
Custom bbox evaluation is contained in `QuadROICocoMetric`; `coco_metric.py`
remains unchanged. If you synced the earlier modified `coco_metric.py`, replace
it once with the restored version from this project.

V2 retains all network shapes, weights, inference operators and source outputs.
It changes ROI sampling, training supervision, the fine-tuning schedule and
evaluation settings. The original config keeps the original loss because
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
- Validation/test use at most eight pseudo ROIs and score threshold 0.001.
  Training and validation share the same grouping and priority rules below.
  Best checkpoints are selected by `coco/bbox_mAP_50`.
- The 0.001 score threshold is for offline evaluation, not a recommended deployed
  decision threshold. Top-k limits, NMS and output caps are unchanged.

## ROI candidates and grouping

`min_gt_area=64` removes GT boxes with VGA width * height < 64 after crop,
resize and clipping, before creating targets or ROI candidates. The area cutoff
retains boxes exactly at 64; it does not require both sides to be at least 8px.
Removed people are treated as background, including in validation: detections
there can count as false positives. `QuadROICocoMetric` filters its in-memory GT
with the same validation VGA mapping and cutoff, preserving original COCO
coordinates, annotation areas, and images without remaining people. The source
annotation JSON is unchanged. Training, validation and test share this setting.

The shared `height_thresholds=(40, 80)` uses person-box height after the VGA
transform: small < 40px, medium 40px to < 80px, large >= 80px. These approximate
>10m, 5-10m and <5m for a standing 1.7m person, a 640x480 image and a 120-degree
diagonal rectilinear field of view. The projected heights at 10m and 5m are about
39px and 79px. COCO has no distance labels; this is a configurable size proxy,
not a depth measurement.

Validation/test `bbox_mAP_s`, `bbox_mAP_m`, `bbox_mAP_l` and the printed size-group
AR now use these VGA height groups, replacing COCO's original area bins. Exactly
40px belongs to medium and exactly 80px belongs to large. The evaluator applies
the same height ranges to GT and unmatched detections. Original boxes still
determine IoU, and overall AP/AP50 retain their existing definition after the
minimum-GT-area filter. The grouped AP/AR still average IoU 0.50:0.95; they are
not AP50-only metrics. Each evaluation logs the configured height boundaries.
Change the shared height thresholds in the v2 config to update ROI sampling and
evaluation together; old COCO size-group results are not directly comparable.

- Training retains keep probabilities `(0.85, 0.65, 0.3)`, 0-3 random candidates,
  whole-frame ROI dropout and label-free sampling.
- Validation/test use keep probabilities `(1, 1, 0)`, 0-2 random candidates and
  no whole-frame or GT-candidate dropout. Large GT boxes never generate their own
  candidates, but may fall inside a random or grouped ROI. All retained GT boxes still
  participate in detection evaluation.
- All candidates, including random ones, use the same deterministic greedy
  grouping: start with the smallest remaining candidate and add the eligible
  candidate with the lowest enclosing-area/summed-member-area ratio.
- After grouping, jitter each enclosing window with independent width/height
  scales and translation, then align all four boundaries to multiples of four.
- Rank grouped ROIs before singletons, then by final area ascending within each
  class, and retain at most eight. Groups of two and three have the same priority.
  All retained small/medium GT boxes enter validation candidates; the final budget can
  still leave some without HD coverage. Validation uses a fixed per-image seed.

The shared `roi_settings.merge_cfg` exposes:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_members` | 3 | Maximum candidates per group |
| `size_ratio` | 1.5 | Maximum/minimum member width and height ratios |
| `gap_ratio` | 0.5 | Gap to any existing member, normalized by the smaller width/height |
| `area_ratio` | 1.5 | Enclosing area / sum of member areas |
| `min_height` | 24 | Minimum estimated member height in the 64px mosaic input |
| `min_center_distance` | 2.0 | Minimum pairwise center distance in the 8x8 head grid, using the larger axis distance |

The last two checks account for the largest configured scale jitter and outward
alignment when estimating resolution. Every added member is checked against the
whole proposed group. `merge_cfg=None` disables grouping. Grouping changes crop
windows only; individual person labels and the network remain unchanged.

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
Existing v2 weights remain compatible; resume after syncing the updated files to
use the new ROI policy. Restart the process to load the updated transform/config.

## Same-checkpoint evaluations

These commands use the existing v1 checkpoint and require no training. Substitute
the actual checkpoint filename if necessary. The eight- and three-ROI v2 variants
use the same per-image seed and sampling parameters; changing the budget truncates the sorted
ROI list. The HD head's new loss option does not affect inference on old weights.

Eight ROI budget, threshold 0.001 (v2's default validation setting):

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi8_low
```

Ambient only, same threshold:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_ambient --cfg-options test_dataloader.dataset.pipeline.2.pseudo_rois=False
```

Optional three-ROI comparison, threshold 0.001:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector_v2.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi3 --cfg-options test_dataloader.dataset.pipeline.2.max_rois=3
```

To reproduce the original ROI policy and threshold 0.05, use the original config:

```powershell
python tools/test.py configs/ambient_train/quad_roi_detector.py work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth --work-dir work_dirs/quad_roi_eval/v1_roi8_original
```

To evaluate a trained v2 model, substitute its checkpoint path in the same
commands. Compare v1 and v2 weights using the same config and ROI policy;
the old logs and updated v2 results use different evaluation settings. Changing
only the top-level `roi_settings` through CLI does not rebuild an already-defined
pipeline; the overrides above target the actual test pipeline transform.

No training, inference, or tests were run while implementing this change.
