# training schedule for 1x
_base_ = [
    '_base_aster.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adamw_cos_6e.py',
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/aster/aster_resnet45_6e_st_mj/aster_resnet45_6e_st_mj-cc56eca4.pth"
# dataset settings
train_list = [
    _base_.OccludedRoadText_textrecog_train,
]
test_list = [
    _base_.OccludedRoadText_textrecog_test,
]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50))

train_dataset = dict(
    type='ConcatDataset', datasets=train_list, pipeline=_base_.train_pipeline)
test_dataset = dict(
    type='ConcatDataset', datasets=test_list, pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=1024,
    num_workers=24,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

auto_scale_lr = dict(base_batch_size=1024)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['RoadText'])
test_evaluator = val_evaluator
