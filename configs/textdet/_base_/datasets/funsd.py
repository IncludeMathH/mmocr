funsd_textdet_data_root = 'data/funsd'

funsd_textdet_train = dict(
    type='OCRDataset',
    data_root=funsd_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

funsd_textdet_test = dict(
    type='OCRDataset',
    data_root=funsd_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)