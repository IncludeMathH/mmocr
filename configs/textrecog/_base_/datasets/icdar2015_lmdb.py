icdar2015_lmdb_textrecog_data_root = 'data/icdar2015'

icdar2015_lmdb_textrecog_train = dict(
    type='RecogLMDBDataset',
    data_root=icdar2015_lmdb_textrecog_data_root,
    ann_file='textrecog_train.lmdb',
    pipeline=None)

icdar2015_lmdb_textrecog_test = dict(
    type='RecogLMDBDataset',
    data_root=icdar2015_lmdb_textrecog_data_root,
    ann_file='textrecog_test.lmdb',
    test_mode=True,
    pipeline=None)

icdar2015_lmdb_1811_textrecog_test = dict(
    type='RecogLMDBDataset',
    data_root=icdar2015_lmdb_textrecog_data_root,
    ann_file='textrecog_test_1811.lmdb',
    test_mode=True,
    pipeline=None)
