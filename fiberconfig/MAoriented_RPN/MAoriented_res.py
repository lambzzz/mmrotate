
###################
# default_runtime #
################### 

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'






###############
# schedule_1x #
############### 

# evaluation
evaluation = dict(interval=10, metric='mAP')
# optimizer
optimizer = dict(type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[40, 50])
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=60)




##################
# model settings #
##################

angle_version = 'le90'
model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='MAOrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='RealRotatedAnchorGenerator',
            angles=[0, 45],
            scales=[6, 8, 10],
            ratios=[0.25, 4.0, 0.125, 8.0, 0.05, 20],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='RotatedMidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),

    train_cfg=dict(                                         #rpn 和 rcnn 训练超参数的配置
        rpn=dict(
            assigner=dict(                                      #分配正负样本分配器的配置
                type='MaxIoUAssigner',                          #选用MaxIoUAssigner
                pos_iou_thr=0.7,                                #IOU >= 0.7 作为正样本
                neg_iou_thr=0.3,                                #IOU <= 0.3 作为负样本
                min_pos_iou=0.3,                                #作为正样本的最小IOU阈值
                match_low_quality=True,                         #是否匹配低质量的框
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),                             #忽略 bbox 的 IoF 阈值
            sampler=dict(                                       #正负采样器的配置
                type='RRandomSampler',                          #选用RandomSampler采样器
                # num=256,
                num=2560,                                       #需要提取样本的数量
                pos_fraction=0.5,                               #正样本占总样本数的比例
                neg_pos_ub=-1,                                  #基于正样本数量的负样本上限，超出上限的忽略，-1表示不忽略
                add_gt_as_proposals=False),                     #采样后是否添加 GT 作为 proposal
            allowed_border=-1,                                  #对有效anchor进行边界填充，-1表示不填充
            pos_weight=-1,                                      #训练期间正样本权重，-1代表不更改
            debug=False),                                       #是否设置调试(debug)模式
        rpn_proposal=dict(                                      #在训练期间生成 proposals 的配置
            # nms_pre=2000,
            # max_per_img=2000,
            nms_pre=8000,                                       #做非极大值抑制(NMS)前box的数量
            max_per_img=8000,                                   #做NMS后要保留的box的数量
            nms=dict(type='nms_rotated', iou_threshold=0.8),    #NMS，其阈值为0.7
            min_bbox_size=0),                                   #允许的最小 box 尺寸
        rcnn=dict(
            assigner=dict(                                  #RCNN分配正负样本
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,                            #IOU >= 0.5 作为正样本
                neg_iou_thr=0.5,                            #IOU < 0.5 作为负样本
                min_pos_iou=0.5,                            #将 box 作为正样本的最小 IoU 阈值
                match_low_quality=False,                    #是否匹配低质量的框
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),                         #忽略 bbox 的 IoF 阈值，-1表示不忽略
            sampler=dict(                                   #正负采样器的配置
                type='RRandomSampler',
                # num=512,
                num=2048,                                   #需要提取样本的数量
                pos_fraction=0.25,                          #正样本占总样本数的比例
                neg_pos_ub=-1,                              #基于正样本数量的负样本上限，超出上限的忽略，-1表示不忽略
                add_gt_as_proposals=True),                  #采样后是否添加 GT 作为 proposal
            pos_weight=-1,                                  #训练期间正样本权重，-1代表不更改
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            # nms_pre=2000,
            # max_per_img=2000,
            nms_pre=8000,
            max_per_img=8000,
            nms=dict(type='nms_rotated', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            # nms_pre=2000,
            nms_pre=8000,
            min_bbox_size=0,
            score_thr=0.05,                                 #bbox的分数阈值
            nms=dict(iou_thr=0.3),
            max_per_img=2000)))                             

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RResize', img_scale=(1024, 1024)),
#     dict(
#         type='RRandomFlip',
#         flip_ratio=[0.25, 0.25, 0.25],
#         direction=['horizontal', 'vertical', 'diagonal'],
#         version=angle_version),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# data = dict(
#     train=dict(pipeline=train_pipeline, version=angle_version),
#     val=dict(version=angle_version),
#     test=dict(version=angle_version))

# optimizer = dict(lr=0.005)







####################
# dataset settings #
####################

# dataset settings
dataset_type = 'DOTADataset'
classes = ['fiber']
data_root = '/root/data/fiber_T2/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'trainval/annfiles/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version=angle_version))




