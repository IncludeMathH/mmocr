_base_ = [
    '_base_mask-rcnn_resnet50_fpn.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_base.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.08))
train_cfg = dict(max_epochs=40, val_interval=1)
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', milestones=[15, 25], end=40),
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
