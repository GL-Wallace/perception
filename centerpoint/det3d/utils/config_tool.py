"""配置修改工具。

提供更改检测范围、推算下采样倍率等函数；部分函数仅适用于 KITTI 数据集的
protobuf 配置格式（model_config）。
"""

from pathlib import Path

import numpy as np
from google.protobuf import text_format


def change_detection_range(model_config, new_range):
    """修改 KITTI 模型配置中的检测范围。

    同步更新 voxel_generator 的点云范围、target_assigner 的 anchor 范围/偏移，
    以及 post_center_limit_range。

    Args:
        model_config: protobuf 模型配置。
        new_range: 形如 [-50, -50, 50, 50] 的列表。
    """
    assert len(new_range) == 4, "you must provide a list such as [-50, -50, 50, 50]"
    old_pc_range = list(model_config.voxel_generator.point_cloud_range)
    old_pc_range[:2] = new_range[:2]
    old_pc_range[3:5] = new_range[2:]
    model_config.voxel_generator.point_cloud_range[:] = old_pc_range
    for anchor_generator in model_config.target_assigner.anchor_generators:
        a_type = anchor_generator.WhichOneof("anchor_generator")
        if a_type == "anchor_generator_range":
            # range 型 anchor：更新 anchor_ranges 的 x/y 范围。
            a_cfg = anchor_generator.anchor_generator_range
            old_a_range = list(a_cfg.anchor_ranges)
            old_a_range[:2] = new_range[:2]
            old_a_range[3:5] = new_range[2:]
            a_cfg.anchor_ranges[:] = old_a_range
        elif a_type == "anchor_generator_stride":
            # stride 型 anchor：根据新范围与步长重新计算各轴偏移。
            a_cfg = anchor_generator.anchor_generator_stride
            old_offset = list(a_cfg.offsets)
            stride = list(a_cfg.strides)
            old_offset[0] = new_range[0] + stride[0] / 2
            old_offset[1] = new_range[1] + stride[1] / 2
            a_cfg.offsets[:] = old_offset
        else:
            raise ValueError("unknown")
    old_post_range = list(model_config.post_center_limit_range)
    old_post_range[:2] = new_range[:2]
    old_post_range[3:5] = new_range[2:]
    model_config.post_center_limit_range[:] = old_post_range


def get_downsample_factor(model_config):
    """推算模型相对原始输入的累计下采样倍率。

    结合 neck 的下采样/上采样步长与 backbone 的 ds_factor 计算得到。
    """
    try:
        neck_cfg = model_config["neck"]
    except:
        model_config = model_config['first_stage_cfg']
        neck_cfg = model_config['neck']
    downsample_factor = np.prod(neck_cfg.get("ds_layer_strides", [1]))
    if len(neck_cfg.get("us_layer_strides", [])) > 0:
        downsample_factor /= neck_cfg.get("us_layer_strides", [])[-1]

    backbone_cfg = model_config['backbone']
    downsample_factor *= backbone_cfg["ds_factor"]
    downsample_factor = int(downsample_factor)
    assert downsample_factor > 0
    return downsample_factor
