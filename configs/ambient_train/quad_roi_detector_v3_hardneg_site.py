_base_ = ['./quad_roi_detector_v3_hardneg.py']

site_dataset = dict(_base_.train_dataloader.dataset)
site_dataset.update(
    data_root='data/quad_roi_site/', ann_file='train.json',
    data_prefix=dict(img='images/'))

train_dataloader = dict(
    batch_size=12,
    sampler=dict(_delete_=True, type='MultiSourceSampler',
                 batch_size=12, source_ratio=[3, 1]),
    dataset=dict(_delete_=True, type='ConcatDataset',
                 datasets=[dict(_base_.train_dataloader.dataset), site_dataset]))

max_iters = 15000
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop',
                 max_iters=max_iters, val_interval=1000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=200),
    dict(type='CosineAnnealingLR', T_max=max_iters - 200, eta_min=0.00001,
         by_epoch=False, begin=200, end=max_iters)]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000),
    logger=dict(log_metric_by_epoch=False))
work_dir = './work_dirs/quad_roi_detector_v3_hardneg_site'
