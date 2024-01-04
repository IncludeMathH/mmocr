_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/sar.py',
    '../../_base_/schedules/schedule_adam_step_5e.py',
    '../../_base_/recog_pipelines/sar_pipeline.py',
    '../../_base_/recog_datasets/roadtext.py'
]

load_from = "https://download.openmmlab.com/mmocr/textrecog/sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth"
optimizer = dict(type='Adam', lr=1e-4)

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

data = dict(
    workers_per_gpu=2,
    samples_per_gpu=8,
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
