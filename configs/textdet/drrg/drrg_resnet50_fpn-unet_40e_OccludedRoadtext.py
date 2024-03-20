_base_ = [
    '_base_drrg_resnet50_fpn-unet.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# modify max epoch
_base_.train_cfg.max_epochs = 40
_base_.train_cfg.val_interval = 1
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-8, end=40),
]

optim_wrapper = dict(
    optimizer=dict(lr=1e-10, momentum=0.9, type='SGD', weight_decay=0.001),
    type='OptimWrapper')

load_from = "https://download.openmmlab.com/mmocr/textdet/drrg/drrg_resnet50_fpn-unet_1200e_ctw1500/drrg_resnet50_fpn-unet_1200e_ctw1500_20220827_105233-d5c702dd.pth"

# dataset settings
roadtext_textdet_train = _base_.occludedRoadtext_det_train
roadtext_textdet_train.pipeline = _base_.train_pipeline
roadtext_textdet_test = _base_.occludedRoadtext_det_test
roadtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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

auto_scale_lr = dict(base_batch_size=16)
