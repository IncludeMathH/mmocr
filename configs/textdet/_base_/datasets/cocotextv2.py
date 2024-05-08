cocotextv2_textdet_data_root = 'data/cocotextv2'

cocotextv2_textdet_train = dict(
    type='OCRDataset',
    data_root=cocotextv2_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

cocotextv2_textdet_test = dict(
    type='OCRDataset',
    data_root=cocotextv2_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
