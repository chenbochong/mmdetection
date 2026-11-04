_base_ = ['./quad_roi_detector.py']

min_gt_area = 64
height_thresholds = (40, 130)

model = dict(
    type='QuadROIDetectorV3',
    backbone=dict(_delete_=True, type='QuadROIBackboneV3',
                  ambient_channels=64, roi_channels=64),
    neck=dict(ambient_channels=64, context_channels=32),
    ambient_head=dict(channels=64, hidden_channels=32,
                      direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    coarse_head=dict(type='QuadROICenterHead', channels=64, hidden_channels=32,
                     direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    head=dict(direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    fused_head=dict(type='QuadROICenterHead', channels=64, box_mode='ltrb',
                    direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    min_visibility=0.25, large_height=height_thresholds[1], support_iof=0.8,
    score_thr=0.001)

roi_settings = dict(
    max_rois=8, min_gt_area=min_gt_area, height_thresholds=height_thresholds,
    matched_negatives=True, negative_attempts=32,
    merge_cfg=dict(
        max_members=3, size_ratio=1.5, gap_ratio=0.5, area_ratio=1.5,
        min_height=24, min_center_distance=2.0))

train_pipeline = [dict(step) for step in _base_.train_pipeline]
train_pipeline[-1].update(roi_settings)
train_pipeline[-1].update(partial_prob=0.3, partial_fraction=(0.3, 0.6))
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_pipeline = [dict(step) for step in _base_.val_pipeline]
val_pipeline[-1].update(roi_settings)
val_pipeline[-1].update(
    keep_probs=(1.0, 0.8, 0.3), random_rois=(0, 2),
    no_roi_prob=0.0, label_free_prob=0.0, partial_prob=0.0)
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

val_evaluator = dict(type='QuadROICocoMetric', min_gt_area=min_gt_area,
                     height_thresholds=height_thresholds, report_ap50_by_size=True)
test_evaluator = dict(type='QuadROICocoMetric', min_gt_area=min_gt_area,
                      height_thresholds=height_thresholds, report_ap50_by_size=True)

default_hooks = dict(checkpoint=dict(save_best='coco/bbox_mAP_50', rule='greater'))

load_from = None
resume = False
work_dir = './work_dirs/quad_roi_detector_v3'
