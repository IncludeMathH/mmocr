roadtext_textrecog_data_root = 'data/roadtext/'

roadtext_textrecog_train = dict(
    type='OCRDataset',
    data_root=roadtext_textrecog_data_root,
    ann_file='train_mmocr-style.json',
    data_prefix=dict(img_path='train_images/'),
    test_mode=False,
    pipeline=None)

roadtext_textrecog_test = dict(
    type='OCRDataset',
    data_root=roadtext_textrecog_data_root,
    ann_file='val_mmocr-style.json',
    data_prefix=dict(img_path='val_images/'),
    test_mode=True,
    pipeline=None)
