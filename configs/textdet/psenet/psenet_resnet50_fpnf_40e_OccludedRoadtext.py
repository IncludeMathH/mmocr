_base_ = [
    '_base_psenet_resnet50_fpnf.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(val_interval=1, max_epochs=40)
param_scheduler = [
    dict(type='MultiStepLR', milestones=[15, 25], end=40),
]

# dataset settings
roadtext_textdet_train = _base_.occludedRoadtext_det_train
roadtext_textdet_test = _base_.occludedRoadtext_det_test

# use quadrilaterals for roadtext
model = dict(
    backbone=dict(style='pytorch'),
    det_head=dict(postprocessor=dict(text_repr_type='quad')))

# pipeline settings
roadtext_textdet_train.pipeline = _base_.train_pipeline
roadtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=roadtext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=roadtext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=64 * 4)
