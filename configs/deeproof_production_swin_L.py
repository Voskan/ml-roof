
# DeepRoof-2026: Production Configuration for A100 Cluster (4 GPUs)
_base_ = [
    './swin/swin_large.py', # Inherit Backbone & Neck from Swin-L
]

# 0. Custom Imports
custom_imports = dict(
    imports=[
        'mmdet.models',
        'deeproof.models.backbones.swin_v2_compat',
        'deeproof.datasets.roof_dataset',
        'deeproof.models.heads.mask2former_head',
        'deeproof.models.heads.dense_normal_head',
        'deeproof.models.heads.edge_head',
        'deeproof.models.deeproof_model',
        'deeproof.models.heads.geometry_head',
        'deeproof.models.losses',
        'deeproof.optim.safe_optim_wrapper',
        'deeproof.evaluation.metrics',
        'deeproof.hooks.progress_hook',
    ],
    allow_failed_imports=False)

# 1. Shared Model Settings
# Expanded 10-class schema (merged from MassiveMasterDataset + roof_information_dataset_2 + yolo_satellite):
#  0=BG, 1=Flat, 2=Sloped-South, 3=SolarPanel, 4=Obstacle-Generic,
#  5=Chimney, 6=Dormer/Skylight/Window, 7=Sloped-North, 8=Sloped-EastWest, 9=AC/Mech
num_classes = 10
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255, # 255 is ignore_index
    test_cfg=dict(size_divisor=32))

# 1. Model Configuration
model = dict(
    type='DeepRoofMask2Former', # Our Multi-Task Model
    data_preprocessor=data_preprocessor,
    test_cfg=dict(
        mode='whole',
        # Lowered from 0.20 to 0.10: obstacles cover only 0.5% of pixels and are small.
        # Too high a threshold silently drops all obstacle predictions.
        instance_score_thr=0.10,
        instance_min_area=32,   # Smaller: obstacles can be tiny (chimneys, vents)
        max_instances=200,      # Match num_queries — some images have 130+ instances
    ),

    # MassiveMasterDataset has no reliable normal maps.
    # Disable geometry branches completely to avoid dead/random supervision paths.
    geometry_head=None,
    geometry_loss=None,
    geometry_loss_weight=0.0,
    dense_geometry_head=None,
    dense_normal_loss=None,
    dense_geometry_loss_weight=0.0,
    piecewise_planar_loss_weight=0.0,
    edge_head=None,
    edge_loss_weight=0.0,  # No edge GT in MassiveMasterDataset
    sam_distill_weight=0.0,  # No SAM masks in MassiveMasterDataset
    topology_loss_weight=0.0,  # Disable until core segmentation converges

    decode_head=dict(
        type='DeepRoofMask2FormerHead',
        in_channels=[192, 384, 768, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        # Dataset analysis: mean 18.78 instances/image, max=130. 100 queries loses many.
        # 200 queries covers 99%+ of images without excessive memory overhead.
        num_queries=200,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_levels=3,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(
                num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type='DeepRoofCrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            # 10-class weights (index=10 is no-object, Mask2Former default=0.1)
            # BG=1.0  Flat=8.0  Sloped-S=3.0  Panel=15.0  Obstacle=15.0
            # Chimney=20.0  Dormer=10.0  Sloped-N=3.0  Sloped-EW=3.0  AC=15.0  no-obj=0.1
            class_weight=[1.0, 8.0, 3.0, 15.0, 15.0, 20.0, 10.0, 3.0, 3.0, 15.0, 0.1]),
        loss_mask=dict(
            type='DeepRoofHybridMaskLoss',
            bce_weight=1.0,
            lovasz_weight=1.0,
            loss_weight=5.0,
            debug_first_n_calls=0,
            reduction='mean'),
        loss_dice=dict(
            type='DeepRoofDiceBoundaryLoss',
            dice_weight=0.8,
            boundary_weight=0.2,
            loss_weight=5.0,
            eps=1e-6,
            reduction='mean',
            use_sigmoid=True),
        train_cfg=dict(
            # Reduced from 12544 to 8192 for 512x512 inputs — still samples 3.1% of pixels.
            # Saves ~35% Hungarian matching time with negligible quality impact.
            num_points=8192,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    # ClassificationCost weight=2.0: class matching drives query specialization.
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    # CrossEntropy mask cost: primary spatial alignment signal.
                    dict(
                        type='mmdet.CrossEntropyLossCost',
                        weight=5.0,
                        use_sigmoid=True),
                    # DiceCost: overlap quality — crucial for getting tight facet boundaries.
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')))
)

# 2. Data Pipeline & Dataloader
dataset_type = 'DeepRoofDataset'
data_root = 'data/MassiveMasterDataset/'

# DeepRoofDataset handles loading and augmentation in its custom __getitem__.
train_pipeline = []

train_dataloader = dict(
    batch_size=8, # Samples per GPU (Total Batch Size will be 8 * num_gpus)
    num_workers=8,
    timeout=300,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        img_suffix='.jpg',
        seg_map_suffix='.png',
        normal_suffix='.npy',
        slope_threshold_deg=2.0,
        min_instance_area_px=16,
        max_instances_per_image=128,
        sr_dual_prob=0.25,
        sr_scale=2.0,
        image_size=(512, 512),  # Match actual dataset resolution
        pipeline=train_pipeline
    )
)

val_pipeline = []

# Test/inference pipeline: standard mmseg pipeline for external images.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='PackSegInputs'),
]

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in [0.75, 1.0, 1.25]
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='PackSegInputs')]
        ])
]

tta_model = dict(type='SegTTAModel')

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    timeout=300,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        test_mode=True,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        normal_suffix='.npy',
        slope_threshold_deg=2.0,
        min_instance_area_px=16,
        max_instances_per_image=128,
        image_size=(512, 512),
        pipeline=val_pipeline
    )
)

val_evaluator = [
    dict(type='IoUMetric', iou_metrics=['mIoU'], prefix=''),
    dict(type='DeepRoofBoundaryMetric', tolerance=1),
    dict(
        type='DeepRoofFacetMetric',
        overlap_threshold=0.30,
        # Thresholds must match model.test_cfg for consistent eval/inference behaviour.
        score_thr=0.10,   # was 0.20 — lowered to capture small obstacles
        min_area=32,      # was 64 — smaller obstacles (chimneys, vents)
        max_dets=200,     # was 120 — match num_queries
    ),
]
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# 3. Optimizer & Scheduler (A100 Best Practice)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999)
)

optim_wrapper = dict(
    type='DeepRoofSafeOptimWrapper',
    optimizer=optimizer,
    skip_nonfinite_grad=True,
    max_nonfinite_warnings=20,
    # Mask2Former standard: max_norm=1.0 keeps gradient scale healthy.
    # 0.01 was 100x too small — it zeroed every update and caused NaN at iter=10.
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0)
        }
    )
)

param_scheduler = [
    # Linear Warmup for 1500 iterations
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    # Poly Decay for the rest of 100k iters
    dict(
        type='PolyLR',
        power=0.9,
        begin=1500,
        end=100000,
        eta_min=1e-6,
        by_epoch=False,
    )
]

# 4. Runtime Config
train_cfg = dict(type='IterBasedTrainLoop', max_iters=100000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
custom_hooks = [
    dict(type='DeepRoofProgressHook', interval=10, heartbeat_sec=20, dataloader_warn_sec=90, flush=True),
]
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False,
    ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        save_best='facet/AP50',
        rule='greater',
    )
)
log_processor = dict(by_epoch=False, window_size=10)
log_level = 'INFO'
