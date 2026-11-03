_base_ = ['../_base_/default_runtime.py']

model = dict(
    type='QuadROIDetector',
    data_preprocessor=dict(type='QuadROIPreprocessor'),
    backbone=dict(type='QuadROIBackbone', ambient_channels=32,
                  roi_channels=64, context_channels=64),
    neck=dict(type='QuadROIFusion', ambient_channels=32, roi_channels=64),
    ambient_head=dict(type='QuadROICenterHead', channels=32),
    head=dict(type='QuadROICenterHead', channels=64),
    min_visibility=0.6,
    score_thr=0.05,
    nms_iou=0.5,
    support_iou=0.5,
    ambient_topk=64,
    roi_topk=16,
    max_per_img=64)

data_root = 'D:/AI-Server/workspace/datasets/coco/'
roi_settings = dict(
    type='QuadROICrop', max_rois=8,
    keep_probs=(0.85, 0.65, 0.3), area_thresholds=(4096, 32768),
    roi_scale=(0.9, 1.4), shift=0.15,
    random_rois=(0, 3), no_roi_prob=0.15, label_free_prob=0.2)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, box_type=None),
    dict(**roi_settings, training=True, crop_scale=(0.6, 1.0))]

# Label-derived validation ROIs simulate the sensor; these are not proposal recall metrics.
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, box_type=None),
    dict(**roi_settings, training=False, seed=2026)]

inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='QuadROICrop', training=False, pseudo_rois=False, max_rois=8)]

train_dataloader = dict(
    batch_size=12, num_workers=8, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        metainfo=dict(classes=('person',)),
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'), metainfo=dict(classes=('person',)),
        test_mode=True, pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

max_epochs = 120
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0.00001,
         by_epoch=True, begin=0, end=max_epochs)]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1,
                                     max_keep_ckpts=3, save_best='coco/bbox_mAP'))
auto_scale_lr = dict(enable=False, base_batch_size=12)
randomness = dict(seed=2026)
find_unused_parameters = True
