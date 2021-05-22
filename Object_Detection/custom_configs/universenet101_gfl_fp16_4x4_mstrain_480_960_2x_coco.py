model = dict(
    type='GFL',
    pretrained=None,
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=False,
            lcconv_padding=1)
    ],
    bbox_head=dict(
        type='GFLSEPCHead',
        num_classes=11,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file='../input/data/train.json',
        img_prefix='../input/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='InstaBoost',
                action_candidate=('normal', 'horizontal', 'skip'),
                action_prob=(1, 0, 0),
                scale=(0.8, 1.2),
                dx=15,
                dy=15,
                theta=(-1, 1),
                color_prob=0.5,
                hflag=False,
                aug_ratio=0.5),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1333, 480), (1333, 960)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    val=dict(
        type='CocoDataset',
        ann_file='../input/data/val.json',
        img_prefix='../input/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')),
    test=dict(
        type='CocoDataset',
        ann_file='../input/data/test.json',
        img_prefix='../input/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(666, 400), (999, 600), (1333, 800), (1666, 1000),
                           (1999, 1200)],
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('UNKNOWN', 'General trash', 'Paper', 'Paper pack', 'Metal',
                 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
                 'Clothing')))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='pstage-3-od',
                name='universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/pstage_3/mmdetection/pretrained/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco.pth'
resume_from = None
workflow = [('train', 1)]
fp16 = None
seed = 42
gpu_ids = [0]
work_dir = './save_dir/universenet101_gfl_fp16_4x4_mstrain_480_960_2x_coco'
