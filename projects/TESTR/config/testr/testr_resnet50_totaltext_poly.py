_base_ = [
    '_base_testr_resnet50.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_500e.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textspotting/abcnet/abcnet_resnet50_fpn_500e_icdar2015/abcnet_resnet50_fpn_pretrain-d060636c.pth'  # noqa

# dataset settings
totaltext_textspotting_train = _base_.totaltext_textspotting_train
totaltext_textspotting_train.pipeline = _base_.train_pipeline
totaltext_textspotting_test = _base_.totaltext_textspotting_test
totaltext_textspotting_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=totaltext_textspotting_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=totaltext_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_imports = dict(imports=['abcnet'], allow_failed_imports=False)
