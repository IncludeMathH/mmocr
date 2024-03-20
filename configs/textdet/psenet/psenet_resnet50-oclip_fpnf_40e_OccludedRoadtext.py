_base_ = [
    'psenet_resnet50_fpnf_40e_OccludedRoadtext.py',
]

load_from = "https://download.openmmlab.com/mmocr/textdet/psenet/psenet_resnet50-oclip_fpnf_600e_icdar2015/psenet_resnet50-oclip_fpnf_600e_icdar2015_20221101_131357-2bdca389.pth"

_base_.optim_wrapper.optimizer.lr = 5e-5      # 1e-4 -> 5e-5 for fine tune

_base_.model.backbone = dict(
    type='CLIPResNet',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/'
        'mmocr/backbone/resnet50-oclip-7ba0c533.pth'))
