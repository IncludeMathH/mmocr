roadtext_det_data_root = 'data/roadtextDet'

roadtext_det_train = dict(
    type='OCRDataset',
    data_root=roadtext_det_data_root,
    ann_file='val.json',
    data_prefix=dict(img_path='val_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

roadtext_det_test = dict(
    type='OCRDataset',
    data_root=roadtext_det_data_root,
    ann_file='val.json',
    data_prefix=dict(img_path='val_images/'),
    test_mode=True,
    pipeline=None)
