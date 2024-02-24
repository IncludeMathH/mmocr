occludedRoadtext_det_train = dict(
    type='OCRDataset',
    data_root='data/ISTD-OC/detection',
    ann_file='text_spotting.json',
    data_prefix=dict(img_path=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

occludedRoadtext_det_test = dict(
    type='OCRDataset',
    data_root='data/Occluded_RoadText',
    ann_file='text_spotting.json',
    data_prefix=dict(img_path='val/'),
    test_mode=True,
    pipeline=None)
