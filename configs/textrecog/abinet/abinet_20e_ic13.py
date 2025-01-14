_base_ = [
    '../_base_/datasets/icdar2013.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_base.py',
    '_base_abinet.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textrecog/abinet/abinet_pretrain-45deac15.pth'  # noqa

optim_wrapper = dict(optimizer=dict(lr=1e-4))
train_cfg = dict(max_epochs=20)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR', end=2, start_factor=0.001,
        convert_to_iter_based=True),
    dict(type='MultiStepLR', milestones=[16, 18], end=20),
]

# dataset settings
# train_dataset = dict(
#     type='ConcatDataset', datasets=[_base_.icdar2013_textrecog_train], pipeline=_base_.train_pipeline)
# test_dataset = dict(
#     type='ConcatDataset', datasets=[_base_.icdar2013_textrecog_test], pipeline=_base_.test_pipeline)
_base_.icdar2013_textrecog_train.pipeline = _base_.train_pipeline
_base_.icdar2013_textrecog_test.pipeline = _base_.test_pipeline
train_dataset = _base_.icdar2013_textrecog_train
test_dataset = _base_.icdar2013_textrecog_test

train_dataloader = dict(
    batch_size=64,     # 192,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = test_dataloader

val_evaluator = dict(
    dataset_prefixes=['IC13'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=192 * 8)

# 不然会报错，有部分字符没有出现在dictionary中
dictionary = dict(
    with_unknown=True)

model = dict(
    decoder=dict(
        dictionary=dictionary,
        max_seq_len=26,
    ),
)
