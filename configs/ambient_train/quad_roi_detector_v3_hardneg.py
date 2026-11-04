_base_ = ['./quad_roi_detector_v3.py']

train_pipeline = [dict(step) for step in _base_.train_pipeline]
train_pipeline[-1].update(hard_negative_slots=2)
train_dataloader = dict(dataset=dict(
    type='QuadROICocoDataset', pipeline=train_pipeline,
    negative_classes=('chair', 'couch', 'potted plant', 'vase',
                      'backpack', 'suitcase', 'teddy bear')))

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs)
optim_wrapper = dict(optimizer=dict(lr=0.0001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', T_max=max_epochs, eta_min=0.00001,
         by_epoch=True, begin=0, end=max_epochs)]

# Supply the trained v3 checkpoint through --cfg-options load_from=...
load_from = None
resume = False
work_dir = './work_dirs/quad_roi_detector_v3_hardneg'
