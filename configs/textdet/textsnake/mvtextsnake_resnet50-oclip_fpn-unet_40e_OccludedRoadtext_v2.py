"""
训练、eval均使用多视图
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

_base_.optim_wrapper.optimizer.lr = 4e-3
_base_.train_cfg.max_epochs=40
_base_.train_cfg.val_interval=1
_base_.param_scheduler[0].end=40
_base_.default_hooks.checkpoint.interval=1    # 
_base_.model.type='MVTextSnake'
_base_.model.data_preprocessor=dict(
        type='MVTextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32)
_base_.model.process_multi_view = dict(
        input_channel=98,
        output_channel=32,
)

# dataset settings
ic15_textdet_train = _base_.icdar2015_textdet_train
ic15_textdet_train.pipeline = [
    dict(type='LoadMVImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True),
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
    dict(
        type='MMOCRCopyPaste',
        object_dir='data/coco/val2017_objects',
        max_num_pasted=5,
    ),
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
    pipeline=[
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1333, 736), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
])

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=ic15_textdet_train)   # OccludedRoadtext_textdet_train)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=OccludedRoadtext_textdet_test)

test_dataloader = val_dataloader

auto_scale_lr = dict(base_batch_size=4)
