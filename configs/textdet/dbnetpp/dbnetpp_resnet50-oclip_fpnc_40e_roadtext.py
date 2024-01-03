_base_ = [
    'dbnetpp_resnet50-dcnv2_fpnc_1200e_roadtext.py',
]

load_from = 'https://download.openmmlab.com/mmocr/textdet/dbnetpp/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015/dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015_20221101_124139-4ecb39ac.pth'

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.num_workers = 8
_base_.optim_wrapper.optimizer.lr = 0.001   # 0.002

# optimizer and learning policy
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', end=10, start_factor=0.001),
    dict(type='PolyLR', power=0.9, eta_min=1e-7, begin=10, end=40),
]
