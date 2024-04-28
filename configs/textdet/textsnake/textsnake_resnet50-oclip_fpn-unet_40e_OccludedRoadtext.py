"""
TODO: fix bug when using textsnake
"""
_base_ = [
    '_base_textsnake_resnet50_fpn-unet.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sgd_1200e.py',
]

# load pretrained checkpoint
load_from = "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500/textsnake_resnet50-oclip_fpn-unet_1200e_ctw1500_20221101_134814-a216e5b2.pth"

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.optim_wrapper.optimizer.lr = 2e-3
_base_.train_cfg.max_epochs=40
_base_.train_cfg.val_interval=1
_base_.param_scheduler[0].end=40

# dataset settings
ic15_textdet_train = _base_.icdar2015_textdet_train
ic15_textdet_train.pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True),
    dict(
        type='MMOCRCopyPaste',
        object_dir='data/cocotextv2/textdet_imgs/imgs',
    ),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='RandomApply',
        transforms=[dict(type='RandomCrop', min_side_ratio=0.3)],
        prob=0.65),
    dict(
        type='RandomRotate',
        max_angle=20,
        pad_with_fixed_color=False,
        use_canvas=True),
    dict(
        type='BoundedScaleAspectJitter',
        long_size_bound=800,
        short_size_bound=480,
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1)),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(type='Resize', scale=800, keep_ratio=True),
            dict(type='SourceImagePad', target_scale=800)
        ],
                    dict(type='Resize', scale=800, keep_ratio=False)],
        prob=[0.4, 0.6]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='FixInvalidPolygon'),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]

OccludedRoadtext_textdet_test = dict(
    type='OCRDataset',
    data_root='data/Occluded_RoadText',
    data_prefix=dict(img_path='val/'),
    ann_file='text_spotting.json',
    test_mode=True,
    pipeline=_base_.test_pipeline)

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=OccludedRoadtext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)
