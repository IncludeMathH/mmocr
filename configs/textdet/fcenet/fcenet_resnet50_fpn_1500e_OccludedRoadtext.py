_base_ = [
    '_base_fcenet_resnet50_fpn.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

optim_wrapper = dict(optimizer=dict(lr=1e-3, weight_decay=5e-4))
train_cfg = dict(max_epochs=1500)
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=1500),
]

# dataset settings
roadtext_textdet_train = _base_.occludedRoadtext_det_train
roadtext_textdet_test = _base_.occludedRoadtext_det_test
roadtext_textdet_train.pipeline = _base_.train_pipeline
roadtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=roadtext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=roadtext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=8)
