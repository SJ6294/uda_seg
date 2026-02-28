# Binary segmentation dataset config for MAS3K

dataset_type = 'CustomDataset'
data_root = 'B:/3_exp/uda/data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, binary_threshold=128),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, binary_threshold=128),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32, pad_val=0, seg_pad_val=255),
    dict(type='RandomFlip', prob=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    result_dir='result',
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MAS3K/fold1/train/high',
        ann_dir='MAS3K/fold1/train/Mask',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MAS3K/fold1/valid/high',
        ann_dir='MAS3K/fold1/valid/Mask',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='MAS3K/fold1/test/high',
        ann_dir='MAS3K/fold1/test/Mask',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=test_pipeline))
