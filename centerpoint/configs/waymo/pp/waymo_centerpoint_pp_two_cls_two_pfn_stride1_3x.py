"""Waymo 数据集 CenterPoint 单阶段 PointPillars 配置（两类别版本）。

采用 PointPillars 主干：PillarFeatureNet 提取柱体特征（num_filters=[64, 64]、不含距离特征），
RPN 融合下/上采样 BEV 特征，CenterHead 输出 VEHICLE/PEDESTRIAN 的中心热图与属性回归。
体素尺寸 0.32x0.32x6.0 米，点云范围 ±74.88 米，单帧输入（nsweeps=1），未启用 GT-AUG。
使用 Adam + OneCycle（峰值 lr=0.003）训练；当前 total_epochs=1、samples_per_gpu=1，适用于快速验证。
"""
import itertools
import logging
from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=2, class_names=['VEHICLE', 'PEDESTRIAN']),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# 训练与测试共用的目标分配器声明：此处仅转发任务列表，具体分配参数见下方 assigner。
target_assigner = dict(
    tasks=tasks,
)

# 模型配置：reader 提取柱体特征，backbone+neck 编码 BEV 特征，bbox_head 输出中心热图与回归。
model = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=5,
        with_distance=False,
        voxel_size=(0.32, 0.32, 6.0),
        pc_range=(-74.88, -74.88, -2, 74.88, 74.88, 4.0),
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[1, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[1, 2, 4],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=128*3,
        tasks=tasks,
        dataset='waymo',
        weight=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (output_channel, num_conv)
    ),
)

# 训练目标分配器：将 GT 目标中心映射到热图网格并构造回归目标（论文 Sec. 3.2）；
# dense_reg=1 表示在 1 倍步长处做密集回归，损失由 bbox_head 的 weight/code_weights 控制（论文 Sec. 3.3）。
assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

# 推理后处理：过滤中心点合法范围，再做 NMS 去重并与置信度阈值比较，最后还原到点云坐标。
test_cfg = dict(
    post_center_limit_range=[-80, -80, -10.0, 80, 80, 10.0],
    # NMS 参数：预选数量、保留数量上限与 IoU 阈值。
    nms=dict(
        nms_pre_max_size=4096,
        nms_post_max_size=500,
        nms_iou_threshold=0.7,
    ),
    score_threshold=0.1,
    pc_range=[-74.88, -74.88],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.32, 0.32]
)


# 数据集配置：Waymo 数据集，nsweeps=1 表示仅使用单帧激光点云。
dataset_type = "WaymoDataset"
nsweeps = 1
data_root = "data/Waymo"


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.78539816, 0.78539816],
    global_scale_noise=[0.95, 1.05],
    db_sampler=None,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

# 体素化参数：裁剪点云范围并划分为固定尺寸的柱体，限制每柱点数与最大柱数。
voxel_generator = dict(
    range=[-74.88, -74.88, -2, 74.88, 74.88, 4.0],
    voxel_size=[0.32, 0.32, 6.0],
    max_points_in_voxel=20,
    max_voxel_num=[32000, 60000], # we only use non-empty voxels. this will be much smaller than max_voxel_num
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl"
val_anno = "data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl"
test_anno = None

# 数据加载器配置：每卡 batch 大小、每卡数据加载线程数，以及 train/val/test 三个数据集实例。
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# 优化器与学习率：Adam 优化器配合 OneCycle 余弦学习率调度（lr_max 为峰值学习率）。
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# 运行时配置：总训练轮数、GPU 数目、分布式后端与日志级别。
total_epochs = 1
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None 
resume_from = None  
workflow = [('train', 1)]
