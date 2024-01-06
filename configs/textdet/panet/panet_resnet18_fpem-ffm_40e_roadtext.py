_base_ = [
    '../_base_/datasets/roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adam_600e.py',
    '_base_panet_resnet18_fpem-ffm.py',
]

# modify max epoch
_base_.train_cfg.max_epochs = 40
_base_.train_cfg.val_interval = 1
# learning rate
param_scheduler = [
    dict(type='PolyLR', power=0.9, end=40),
]
load_from = "https://download.openmmlab.com/mmocr/textdet/panet/panet_resnet18_fpem-ffm_600e_icdar2015/panet_resnet18_fpem-ffm_600e_icdar2015_20220826_144817-be2acdb4.pth"
_base_.optim_wrapper.optimizer.lr = 5e-4     # 1e-3 -> 5e-4

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=20), )

# dataset settings
roadtext_textdet_train = _base_.roadtext_det_train
roadtext_textdet_test = _base_.roadtext_det_test
# pipeline settings
roadtext_textdet_train.pipeline = _base_.train_pipeline
roadtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=roadtext_textdet_train)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=roadtext_textdet_test)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HmeanIOUMetric', pred_score_thrs=dict(start=0.3, stop=1, step=0.05))
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64)
