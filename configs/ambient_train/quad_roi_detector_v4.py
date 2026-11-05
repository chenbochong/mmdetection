_base_ = ['./quad_roi_detector_v3_hardneg.py']

custom_imports = dict(imports=['mmdet.models.detectors.quad_roi_detector_v4'], allow_failed_imports=False)
model = dict(
    type='QuadROIDetectorV4',
    backbone=dict(_delete_=True, type='QuadROIBackboneV4'),
    neck=dict(ambient_channels=64, context_channels=32, roi_channels=128),
    ambient_head=dict(_delete_=True, type='QuadROIHeadV4', channels=64),
    coarse_head=dict(_delete_=True, type='QuadROIHeadV4', channels=64),
    head=dict(_delete_=True, type='QuadROIHeadV4', channels=128),
    fused_head=dict(_delete_=True, type='QuadROIHeadV4', channels=128, box_mode='ltrb'),
    verifier=dict(type='QuadROIVerifier'), verifier_warmup=2000,
    verifier_samples=8, pre_verify_topk=128, score_thr=0.001)

train_pipeline = [dict(step) for step in _base_.train_pipeline]
train_pipeline[-1].update(roi_size=96)
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_pipeline = [dict(step) for step in _base_.val_pipeline]
val_pipeline[-1].update(roi_size=96)
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))
test_dataloader = val_dataloader

max_epochs = 80
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0.00001,
         by_epoch=True, begin=0, end=max_epochs)]
load_from = None
resume = False
work_dir = './work_dirs/quad_roi_detector_v4'
