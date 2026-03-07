_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/segformer_b5.py',
    './_base_/datasets/uda_mas3k_to_deepfish_sota512.py',
    '../_base_/uda/dacs.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

seed = 0

# 2-class segmentation
model = dict(decode_head=dict(num_classes=2))

# MIC settings (enabled)
uda = dict(
    alpha=0.99,
    pseudo_threshold=0.968,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1.0,
    mask_generator=dict(
        type='block',
        mask_ratio=0.7,
        mask_block_size=64,
        _delete_=True,
    ),
)

optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'segformer_b5_mas3k_to_deepfish_mic_sota'
exp = 'custom_mic_uda'
