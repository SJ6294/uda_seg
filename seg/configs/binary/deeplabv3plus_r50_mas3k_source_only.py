_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/mas3k_512x512.py',
    '../_base_/uda/dacs_srconly.py',
    '../_base_/schedules/sgd.py',
    '../_base_/schedules/poly10.py'
]

seed = 0
model = dict(decode_head=dict(num_classes=2))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'deeplabv3plus_r50_mas3k_source_only'
exp = 'binary_baseline'
