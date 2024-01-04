with_bezier = True

dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../dicts/abcnet.txt',
    with_start=False,
    with_end=False,
    same_start_end=False,
    with_padding=True,
    with_unknown=True)

model = dict(
    detector=dict(
        type='TESTR',
        backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        bbox_head=dict(
            type='TextSpotDETRHead',
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            num_proposals=100,
            pos_embed_scale=1.0,
            num_ctrl_points=16,
            num_classes=1,
            max_text_len=100,
            voc_size=46,
            use_polygon=True,
            aux_loss=False,
            loss_cfg=dict(point_class_weight=1.0,
                        point_coord_weight=1.0,
                        point_text_weight=4.0,
                        box_coord_weight=1.0,
                        box_giou_weight=1.0,
                        box_class_weight=1.0,
                        focal_alpha=0.25,
                        focal_gamma=2.0,
                        aux_loss=False),
            test_score_threshold=0.45,)
    )
)

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(2000, 4000), keep_ratio=True, backend='pillow'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(type='RemoveIgnored'),
    dict(type='RandomCrop', min_side_ratio=0.1),
    dict(
        type='RandomRotate',
        max_angle=30,
        pad_with_fixed_color=True,
        use_canvas=True),
    dict(
        type='RandomChoiceResize',
        scales=[(980, 2900), (1044, 2900), (1108, 2900), (1172, 2900),
                (1236, 2900), (1300, 2900), (1364, 2900), (1428, 2900),
                (1492, 2900)],
        keep_ratio=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
