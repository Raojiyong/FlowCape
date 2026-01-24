log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=50)
evaluation = dict(
    interval=10,
    metric=['PCK', 'NME', 'AUC', 'EPE'],
    key_indicator='PCK',
    gpu_collect=True,
    res_folder='')
optimizer = dict(
    type='Adam',
    lr=1e-5,
)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[160, 180])
total_epochs = 200
# total_epochs = 1
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        [
            0,
        ],
    ],
    inference_channel=[
        0,
    ],
    max_kpt_num=100)

# model settings
model = dict(
    type='FlowPoseModel',
    pretrained='pretrained/swinv2_small_1k_500k.pth',
    # pretrained="swinv2_base",
    text_pretrained='pretrained/Alibaba-NLP/gte-base-en-v1.5',
    finetune_text_pretrained=False,
    align_cfg=dict(
        enable=False,  # Not used in Rectified Flow (x0 = centered support pose)
    ),
    encoder_config=dict(
        type='SwinTransformerV2',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        drop_path_rate=0.3,
        img_size=256,
        upsample="bilinear"
    ),
    keypoint_head=dict(
        type='RiemannianPoseHead',
        img_in_channels=768,
        text_in_channels=768,
        hidden_dim=256,
        time_embed_dim=128,
        with_heatmap=True,
        heatmap_size=64,
        dropout=0.1),
    rfm_cfg=dict(
        # Flow ODE solver settings
        num_timesteps=30,
        t_epsilon=1e-3,
        guidance_scale=0.0,
        atol=1e-5,
        rtol=1e-5,
    ),
    # training and testing settings
    train_cfg=dict(
        with_ode_loss=True,  # End-to-end ODE loss for train/test consistency
        with_heatmap_loss=False,
    ),
    test_cfg=dict(
        use_flow_ode=True,
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11,
        # Keyness guidance (optional, disabled by default for Rectified Flow)
        use_keyness_guidance=False,
        keyness_lambda=0.1,
        use_keyness_refine=False,
        keyness_refine_steps=3,
        keyness_refine_lambda=0.05,
    ))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15,
        scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    # Emit flow targets (x0, x1, mask) instead of Gaussian heatmaps
    dict(type='GenerateFlowTargets'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id', 'skeleton',
            'align_weight',
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='GenerateFlowTargets'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'category_id',
            'skeleton',
            'align_weight',
        ]),
]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
    # samples_per_gpu=16,
    # workers_per_gpu=16,
    # samples_per_gpu=45,
    samples_per_gpu=16,
    workers_per_gpu=16,
    # samples_per_gpu=8,
    # workers_per_gpu=8,
    train=dict(
        type='TransformerFlowPoseDataset',
        ann_file=f'{data_root}/annotations_graph/mp100_split1_train.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerFlowPoseDataset',
        ann_file=f'{data_root}/annotations_graph/mp100_split1_val.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        pipeline=valid_pipeline),
    test=dict(
        type='TestFlowPoseDataset',
        ann_file=f'{data_root}/annotations_graph/mp100_split1_test.json',
        img_prefix=f'{data_root}/images/',
        # img_prefix=f'{data_root}',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=200,
        pck_threshold_list=[0.05, 0.10, 0.15, 0.2, 0.25],
        pipeline=test_pipeline),
)
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

shuffle_cfg = dict(interval=1)
