_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_icdar2015.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth'

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.num_workers = 2
_base_.train_dataloader.batch_size = 3
_base_.train_dataloader.dataset.pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_polygon=True,
        with_label=True,
    ),
    dict(type='MMOCR2MMDet', poly2mask=True),
    dict(
        type='mmdet.RandomErasing',
        n_patches=(5, 10), 
        ratio=(0, 0.2),
        bbox_erased_thr=0.9,
    ),
    dict(type='MMDet2MMOCR'),
    dict(type='FixInvalidPolygon'),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(
        type='ImgAugWrapper',
        args=[['Fliplr', 0.5],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640)),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape'))
]
_base_.val_dataloader.num_workers = 2
_base_.val_dataloader.batch_size = 2
_base_.val_dataloader.dataset.datasets = [
    dict(
        type='OCRDataset',
        data_root='data/Occluded_RoadText',
        data_prefix=dict(img_path='val/'),
        ann_file='text_spotting.json',
        test_mode=True,
        pipeline=None),
]
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.test_dataloader = _base_.val_dataloader
_base_.optim_wrapper.optimizer.lr = 0.001
_base_.train_cfg.max_epochs=40
_base_.train_cfg.val_interval=1

param_scheduler = [
    dict(type='LinearLR', end=10, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=10, end=40),
]
