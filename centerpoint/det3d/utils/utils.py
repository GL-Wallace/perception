"""通用训练工具函数。

提供把数据样本（example）搬运到指定设备的辅助函数，以及 dataloader worker
的随机数种子初始化函数。
"""

import numpy as np
import torch


def example_to_device(
    example, dtype=torch.float32, device=None, non_blocking=True
) -> dict:
    """把一个样本 dict 中的张量搬到指定设备。

    对 anchors/reg_targets 等以 dict 组织的多任务目标，逐项 unsqueeze 后拼接；
    calib 键单独处理为 dict；其余键直接搬运或原样返回。

    Args:
        example: 输入样本 dict。
        dtype: 目标数据类型（本函数当前未显式转换 dtype，仅保留参数）。
        device: 目标设备，默认为 cuda:0。
        non_blocking: 是否异步拷贝。

    Returns:
        dict: 搬运到设备后的样本 dict。
    """
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "reg_targets", "reg_weights", "labels", "anchors_mask"]:
            # 多任务字典型目标：逐项增加 batch 维后拼接。
            res = []
            for kk, vv in v.items():
                vv = [vvv.unsqueeze_(0) for vvv in vv]
                res.append(torch.cat(vv, dim=0).cuda(device, non_blocking=non_blocking))
            example_torch[k] = res
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "points",
            "num_voxels",
        ]:
            # 直接搬运张量；直接提供 fp32 数据配合 dtype=torch.half 会较慢。
            example_torch[k] = v.cuda(device, non_blocking=non_blocking)
        elif k == "calib":
            # 标定数据以 dict 组织，逐项搬运。
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.cuda(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def _worker_init_fn(worker_id):
    """初始化 dataloader worker 的随机数种子（基于时间 + worker_id）。"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
