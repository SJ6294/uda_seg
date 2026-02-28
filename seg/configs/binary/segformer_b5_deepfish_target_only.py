_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b5.py',
    '../_base_/datasets/deepfish_512x512.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0
model = dict(decode_head=dict(num_classes=2))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'segformer_b5_deepfish_target_only'
exp = 'binary_baseline'
