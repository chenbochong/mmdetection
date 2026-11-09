_base_ = ['../_base_/default_runtime.py']

custom_imports = dict(imports=['mmdet.models.detectors.quad_roi_detector_v5'],
                      allow_failed_imports=False)
model = dict(type='QuadROIDetectorV5',
             data_preprocessor=dict(type='QuadROIPreprocessor'),
             strides=(8, 16, 32), regress_ranges=((-1, 64), (64, 128), (128, 1e8)))

data_root = 'D:/AI-Server/workspace/datasets/coco/'
min_gt_area = 64
height_thresholds = (40, 130)
roi_settings = dict(
    type='QuadROICrop', roi_size=96, max_rois=8, min_gt_area=min_gt_area,
    height_thresholds=height_thresholds, roi_scale=(0.9, 1.4), shift=0.15,
    matched_negatives=True, negative_attempts=32,
    merge_cfg=dict(max_members=3, size_ratio=1.5, gap_ratio=0.5, area_ratio=1.5,
                   min_height=24, min_center_distance=2.0))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, box_type=None),
    dict(**roi_settings, training=True, crop_scale=(0.6, 1.0),
         keep_probs=(0.85, 0.65, 0.3), random_rois=(0, 3),
         no_roi_prob=0.15, label_free_prob=0.2, hard_negative_slots=2,
         partial_prob=0.3, partial_fraction=(0.3, 0.6))]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, box_type=None),
    dict(**roi_settings, training=False, seed=2026,
         keep_probs=(1.0, 0.8, 0.3), random_rois=(0, 2),
         no_roi_prob=0.0, label_free_prob=0.0, partial_prob=0.0)]
inference_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='QuadROICrop', roi_size=96, max_rois=8, training=False, pseudo_rois=False)]

train_dataloader = dict(
    batch_size=12, num_workers=8, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='QuadROICocoDataset', data_root=data_root,
                 ann_file='annotations/instances_train2017.json',
                 data_prefix=dict(img='train2017/'), metainfo=dict(classes=('person',)),
                 filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline,
                 negative_classes=('chair', 'couch', 'potted plant', 'vase',
                                   'backpack', 'suitcase', 'teddy bear')))
val_dataloader = dict(
    batch_size=4, num_workers=2, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file='annotations/instances_val2017.json',
                 data_prefix=dict(img='val2017/'), metainfo=dict(classes=('person',)),
                 test_mode=True, pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='QuadROICocoMetric',
                     ann_file=data_root + 'annotations/instances_val2017.json', metric='bbox',
                     min_gt_area=min_gt_area, height_thresholds=height_thresholds,
                     report_ap50_by_size=True)
test_evaluator = val_evaluator

max_epochs = 80
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0.00001,
         by_epoch=True, begin=0, end=max_epochs)]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                                     save_best='coco/bbox_mAP_50', rule='greater'))
auto_scale_lr = dict(enable=False, base_batch_size=12)
randomness = dict(seed=2026)
find_unused_parameters = True
work_dir = './work_dirs/quad_roi_detector_v5'
