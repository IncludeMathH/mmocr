_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_models/robust_scanner.py',
    '../../_base_/schedules/schedule_adam_step_5e.py',
    '../../_base_/recog_pipelines/sar_pipeline.py',
    '../../_base_/recog_datasets/roadtext.py',
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/robustscanner/robustscanner_r31_academic-5f05874f.pth"
optimizer = dict(type='Adam', lr=1e-4)

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
