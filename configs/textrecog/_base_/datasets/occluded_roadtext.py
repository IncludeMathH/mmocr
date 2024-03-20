OccludedRoadText_textrecog_data_root = 'data/Occluded_RoadText'

OccludedRoadText_textrecog_train = dict(
    type='OCRDataset',
    data_root='data/ISTD-OC/recognition',
    ann_file='text_recog.json',
    pipeline=None)

OccludedRoadText_textrecog_test = dict(
    type='OCRDataset',
    data_root=OccludedRoadText_textrecog_data_root,
    data_prefix=dict(img_path='val_recog/'),
    ann_file='text_recog.json',
    test_mode=True,
    pipeline=None)
