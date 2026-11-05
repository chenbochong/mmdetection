_base_ = ['./quad_roi_detector_v4.py']
custom_imports = dict(imports=['mmdet.models.detectors.quad_roi_detector_v4',
                              'mmdet.evaluation.metrics.quad_roi_scene_metric'], allow_failed_imports=False)
scene_pipeline = [dict(step) for step in _base_.val_pipeline]
scene_pipeline[-1].update(hard_negative_slots=2)
val_dataloader = dict(dataset=dict(
    type='QuadROICocoDataset', data_root='data/quad_roi_site/',
    ann_file='val.json', data_prefix=dict(img='images/'), pipeline=scene_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(_delete_=True, type='QuadROISceneMetric')
test_evaluator = val_evaluator
