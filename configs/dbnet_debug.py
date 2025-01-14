auto_scale_lr = dict(base_batch_size=16)
default_hooks = dict(
    checkpoint=dict(interval=20, type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=False,
        draw_pred=False,
        enable=False,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
icdar2015_textdet_data_root = 'data/icdar2015'
icdar2015_textdet_test = dict(
    ann_file='textdet_test.json',
    data_root='data/icdar2015',
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(keep_ratio=True, scale=(
            4068,
            1024,
        ), type='Resize'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
            type='PackTextDetInputs'),
    ],
    test_mode=True,
    type='OCRDataset')
icdar2015_textdet_train = dict(
    ann_file='textdet_train.json',
    data_root='data/icdar2015',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=[
        dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
            with_polygon=True),
        dict(
            brightness=0.12549019607843137,
            op='ColorJitter',
            saturation=0.5,
            type='TorchVisionWrapper'),
        dict(
            args=[
                [
                    'Fliplr',
                    0.5,
                ],
                dict(cls='Affine', rotate=[
                    -10,
                    10,
                ]),
                [
                    'Resize',
                    [
                        0.5,
                        3.0,
                    ],
                ],
            ],
            type='ImgAugWrapper'),
        dict(min_side_ratio=0.1, type='RandomCrop'),
        dict(keep_ratio=True, scale=(
            640,
            640,
        ), type='Resize'),
        dict(size=(
            640,
            640,
        ), type='Pad'),
        dict(
            meta_keys=(
                'img_path',
                'ori_shape',
                'img_shape',
            ),
            type='PackTextDetInputs'),
    ],
    type='OCRDataset')
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmocr/backbone/resnet50-oclip-7ba0c533.pth',
            type='Pretrained'),
        type='CLIPResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        in_channels=256,
        module_loss=dict(type='DBModuleLoss'),
        postprocessor=dict(text_repr_type='quad', type='DBPostprocessor'),
        type='DBHead'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        lateral_channels=256,
        type='FPNC'),
    type='DBNet')
optim_wrapper = dict(
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(end=100, start_factor=0.001, type='LinearLR'),
    dict(begin=100, end=1200, eta_min=1e-07, power=0.9, type='PolyLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                4068,
                1024,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
test_pipeline = [
    dict(color_type='color_ignore_orientation', type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        4068,
        1024,
    ), type='Resize'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
train_cfg = dict(max_epochs=1200, type='EpochBasedTrainLoop', val_interval=20)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='data/icdar2015',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                type='MMOCRCopyPaste',
                object_dir='data/cocotextv2/textdet_imgs/imgs',
            ),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(size=(
                640,
                640,
            ), type='Pad'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                ),
                type='PackTextDetInputs'),
        ],
        type='OCRDataset'),
    num_workers=24,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                4068,
                1024,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

