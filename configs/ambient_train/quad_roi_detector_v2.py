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

val_pipeline = [dict(step) for step in _base_.val_pipeline]
val_pipeline[-1]['max_rois'] = 3
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = dict(dataset=dict(pipeline=val_pipeline))

# Start a new fine-tuning schedule from learned weights, not the old optimizer.
load_from = 'work_dirs/quad_roi_detector/best_coco_bbox_mAP_epoch_37.pth'
resume = False
work_dir = './work_dirs/quad_roi_detector_v2'
