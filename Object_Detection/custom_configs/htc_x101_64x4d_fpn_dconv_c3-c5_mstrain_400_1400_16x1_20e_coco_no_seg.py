dataset_type = "CocoDataset"
data_root = "data/coco/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=False),
    dict(type="Resize", img_scale=[(1200, 300), (1200, 1050)], multiscale_mode="range", keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="CocoDataset",
        ann_file="../input/data/train.json",
        img_prefix="../input/data",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True, with_seg=False),
            dict(type="Resize", img_scale=[(1333, 480), (1333, 960)], multiscale_mode="range", keep_ratio=True),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
        ],
        classes=("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    ),
    val=dict(
        type="CocoDataset",
        ann_file="../input/data/val.json",
        img_prefix="../input/data",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=[(666, 400), (999, 600), (1333, 800), (1666, 1000), (1999, 1200)],
                flip=True,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip", flip_ratio=0.5),
                    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    ),
    test=dict(
        type="CocoDataset",
        ann_file="../input/data/test.json",
        img_prefix="../input/data",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=[(666, 400), (999, 600), (1333, 800), (1666, 1000), (1999, 1200)],
                flip=True,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip", flip_ratio=0.5),
                    dict(type="Normalize", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
        classes=("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"),
    ),
)
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
optimizer = dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=0.001, min_lr=1e-06)
runner = dict(type="EpochBasedRunner", max_epochs=20)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", init_kwargs=dict(project="pstage-3-od", name="htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_c")),
    ],
)
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = "/opt/ml/pstage_3/mmdetection/pretrained/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.pth"
resume_from = None
workflow = [("train", 1)]
model = dict(
    type="HybridTaskCascade",
    pretrained=None,
    backbone=dict(
        type="ResNeXt",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        groups=64,
        base_width=4,
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(type="FPN", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=0.1111111111111111, loss_weight=1.0),
    ),
    roi_head=dict(
        type="HybridTaskCascadeRoIHead",
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor", roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0), out_channels=256, featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
            dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=11,
                bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
            ),
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor", roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0), out_channels=256, featmap_strides=[4, 8, 16, 32]
        ),
        mask_head=[
            dict(
                type="HTCMaskHead",
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
            dict(
                type="HTCMaskHead",
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=11,
                loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
            ),
        ],
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, ignore_iof_thr=-1),
            sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.7, min_pos_iou=0.7, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
        ],
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.001, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100, mask_thr_binary=0.5),
    ),
)
seed = 42
gpu_ids = [0]
work_dir = "./save_dir/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_no_seg"
