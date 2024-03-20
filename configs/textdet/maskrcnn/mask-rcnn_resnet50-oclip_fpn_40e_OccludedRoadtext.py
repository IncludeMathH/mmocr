_base_ = [
    'mask-rcnn_resnet50_fpn_40e_OccludedRoadtext.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/maskrcnn/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015/mask-rcnn_resnet50-oclip_fpn_160e_icdar2015_20221101_131357-a19f7802.pth"

_base_.model.cfg.backbone = dict(
    _scope_='mmocr',
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))

_base_.optim_wrapper.optimizer.lr = 0.01
