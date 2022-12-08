%%writefile {repo_dir+'/local_configs/_base_/datasets/lemons.py'}
# dataset settings
dataset_type = 'CustomDataset'
data_root = '/kaggle/input/lemons-supervisely'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
seg_pad_val = 255

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomRotate', prob=0.75, degree=45, seg_pad_val=seg_pad_val),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.75, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
#     dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=seg_pad_val),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

classes = ['bg','kiwi','lemon']
palette = [[128,0,128], [0,255,0], [200,200,0]]
# classes = ['kiwi','lemon']
# palette = [[0,255,0], [200,200,0]]
reduce_zero_label = False

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='217535_lemons/ds1/img',
        ann_dir='217535_lemons/ds1/masks_machine',
        img_suffix='.jpeg',
        seg_map_suffix='.png',
        pipeline=train_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=reduce_zero_label),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='217535_lemons/ds1/img',
        ann_dir='217535_lemons/ds1/masks_machine',
        img_suffix='.jpeg',
        seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=reduce_zero_label,),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='217792_lemons_test/ds1/img',
#         ann_dir='217535_lemons/ds1/masks_machine',
        img_suffix='.jpeg',
#         seg_map_suffix='.png',
        pipeline=test_pipeline,
        classes=classes,
        palette=palette,
        reduce_zero_label=reduce_zero_label,),
)