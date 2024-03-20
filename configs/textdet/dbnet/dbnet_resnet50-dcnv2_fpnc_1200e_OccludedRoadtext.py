_base_ = [
    '_base_dbnet_resnet50-dcnv2_fpnc.py',
    '../_base_/datasets/occluded_roadtext.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# TODO: Replace the link
load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnet/tmp_1.0_pretrain/dbnet_r50dcnv2_fpnc_sbn_2e_synthtext_20210325-ed322016.pth'  # noqa

# dataset settings
roadtext_textdet_train = _base_.occludedRoadtext_det_train
roadtext_textdet_train.pipeline = _base_.train_pipeline
roadtext_textdet_test = _base_.occludedRoadtext_det_test
roadtext_textdet_test.pipeline = _base_.test_pipeline

train_dataloader = dict(
    batch_size=16,
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

auto_scale_lr = dict(base_batch_size=16)
