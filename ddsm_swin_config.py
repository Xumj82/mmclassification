_base_ = [
    'configs/_base_/models/swin_transformer/tiny_224.py',
    'configs/_base_/schedules/imagenet_bs1024_adamw_swin.py',
    # 'configs/_base_/datasets/imagenet_bs64_swin_224.py',
    'configs/_base_/default_runtime.py'
]


pretrained = 'checkpoints/swin_small_patch4_window7_224-cc7a01c9.pth'
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        arch='tiny', img_size=224),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=3,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=3, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=3, prob=0.5)
    ]))

# dataset settings
dataset_type = 'DdsmPatch'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadMMImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
# train_pipeline = [
#     dict(type='LoadMMImageFromFile'),
#     dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 15)),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     dict(type='RandomFlip', flip_ratio=0.5,direction='vertical'),
#     dict(
#         type='Translate',
#         magnitude_key='magnitude',
#         magnitude_range=(0, 0.2),
#         direction='horizontal'),
#     dict(
#         type='Translate',
#         magnitude_key='magnitude',
#         magnitude_range=(0, 0.2),
#         direction='vertical'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]

# test_pipeline = [
#     dict(type='LoadMMImageFromFile'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/train',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/train_meta.csv',
        pipeline=train_pipeline),
    val=dict(
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/test',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        classes = ('bkg','calc','mass'),
        type=dataset_type,
        data_prefix='/home/xumingjie/dataset/patch_set/img_dir/test',
        ann_file='/home/xumingjie/dataset/patch_set/img_dir/test_meta.csv',
        pipeline=test_pipeline))

evaluation = dict(interval=10, metric='accuracy')
