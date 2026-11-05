_base_ = ['./quad_roi_detector_v4.py']

site_dataset = dict(_base_.train_dataloader.dataset)
site_dataset.update(data_root='data/quad_roi_site/', ann_file='train.json', data_prefix=dict(img='images/'))
train_dataloader = dict(
    batch_size=12,
    sampler=dict(_delete_=True, type='MultiSourceSampler', batch_size=12, source_ratio=[3, 1]),
    dataset=dict(_delete_=True, type='ConcatDataset',
                 datasets=[dict(_base_.train_dataloader.dataset), site_dataset]))
max_iters = 80000
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iters, val_interval=2000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=max_iters - 500, eta_min=0.00001,
         by_epoch=False, begin=500, end=max_iters)]
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000), logger=dict(log_metric_by_epoch=False))
log_processor = dict(by_epoch=False)
work_dir = './work_dirs/quad_roi_detector_v4_site'
