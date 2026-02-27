# Binary UDA dataset config: MAS3K (source) -> Deepfish (target)

dataset_type = 'CustomDataset'
data_root = 'data'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
crop_size = (512, 512)

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, binary_threshold=128),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', binary_label=True, binary_threshold=128),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='MAS3K/train/high',
            ann_dir='MAS3K/train/Mask',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=('background', 'object'),
            palette=[[0, 0, 0], [255, 255, 255]],
            pipeline=source_train_pipeline),
        target=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='Deepfish/train/high',
            ann_dir='Deepfish/train/Mask',
            img_suffix='.jpg',
            seg_map_suffix='.png',
            classes=('background', 'object'),
            palette=[[0, 0, 0], [255, 255, 255]],
            pipeline=target_train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Deepfish/valid/high',
        ann_dir='Deepfish/valid/Mask',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='Deepfish/valid/high',
        ann_dir='Deepfish/valid/Mask',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]],
        pipeline=test_pipeline))
