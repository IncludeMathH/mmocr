_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/datasets/roadtext.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/crnn/crnn_mini-vgg_5e_mj/crnn_mini-vgg_5e_mj_20220826_224120-8afbedbb.pth"

# dataset settings
train_list = [_base_.roadtext_textrecog_train]
test_list = [_base_.roadtext_textrecog_test]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
test_dataloader = val_dataloader

_base_.model.decoder.dictionary.update(
    dict(with_unknown=True, unknown_token=None))
_base_.train_cfg.update(dict(max_epochs=200, val_interval=10))

val_evaluator = dict(dataset_prefixes=['RoadText'])
test_evaluator = val_evaluator
