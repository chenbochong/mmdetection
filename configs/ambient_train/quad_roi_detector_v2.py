_base_ = ['./quad_roi_detector.py']

model = dict(
    ambient_head=dict(direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    head=dict(direct_regression=True, offset_weight=1.0, wh_weight=0.1),
    score_thr=0.001)

max_epochs = 15
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.0002))
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0.00001,
         by_epoch=True, begin=0, end=max_epochs)]

roi_settings = dict(
    max_rois=8,
    height_thresholds=(40, 80),  # Approx. 10m / 5m, 120-degree diagonal FOV, 1.7m person.
    merge_cfg=dict(
        max_members=3, size_ratio=1.5, gap_ratio=0.5, area_ratio=1.5,
        min_height=24, min_center_distance=2.0))  # 64px input / 8x8 head units.

train_pipeline = [dict(step) for step in _base_.train_pipeline]
train_pipeline[-1].update(roi_settings)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

val_pipeline = [dict(step) for step in _base_.val_pipeline]
val_pipeline[-1].update(roi_settings)
val_pipeline[-1].update(
    keep_probs=(1.0, 1.0, 0.0), random_rois=(0, 2),
    no_roi_prob=0.0, label_free_prob=0.0)
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

default_hooks = dict(checkpoint=dict(save_best='coco/bbox_mAP_50', rule='greater'))

# Start a new fine-tuning schedule from learned weights, not the old optimizer.
load_from = 'work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth'
resume = False
work_dir = './work_dirs/quad_roi_detector_v2'
