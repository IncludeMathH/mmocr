_base_ = [
    'dbnet_resnet50-dcnv2_fpnc_1200e_OccludedRoadtext.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_resnet50-oclip_1200e_icdar2015/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth"

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.num_workers = 8
_base_.train_dataloader.batch_size = 24
_base_.optim_wrapper.optimizer.lr = 0.001

param_scheduler = [
    dict(type='LinearLR', end=100, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=100, end=1200),
]

# optimizer and learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=40),
]
