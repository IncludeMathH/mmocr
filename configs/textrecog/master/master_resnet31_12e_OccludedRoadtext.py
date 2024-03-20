_base_ = [
    '_base_master_resnet31.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/master/master_resnet31_12e_st_mj_sa/master_resnet31_12e_st_mj_sa_20220915_152443-f4a5cabc.pth'

optim_wrapper = dict(optimizer=dict(lr=4e-4))
train_cfg = dict(max_epochs=12)
# learning policy
param_scheduler = [
    dict(type='LinearLR', end=100, by_epoch=False),
    dict(type='MultiStepLR', milestones=[11], end=12),
]

# dataset settings
train_list = [_base_.OccludedRoadText_textrecog_train]
test_list = [_base_.OccludedRoadText_textrecog_test]

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

test_dataloader = val_dataloader

val_evaluator = dict(dataset_prefixes=['OccludedRoadText'])
test_evaluator = val_evaluator
