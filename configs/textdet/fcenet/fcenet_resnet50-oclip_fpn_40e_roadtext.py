_base_ = [
    'fcenet_resnet50_fpn_1500e_roadtext.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/fcenet/fcenet_resnet50-oclip_fpn_1500e_icdar2015/fcenet_resnet50-oclip_fpn_1500e_icdar2015_20221101_150145-5a6fc412.pth"

_base_.model.backbone = dict(
    type='CLIPResNet',
    out_indices=(1, 2, 3),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.train_dataloader.batch_size = 16
_base_.train_dataloader.num_workers = 24
_base_.optim_wrapper.optimizer.lr = 0.0002     # 5e-4
_base_.train_cfg.max_epochs = 40
# learning policy
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-7, end=40),
]
