"""批次数据整理（collate）工具。

在 default_collate 基础上扩展对 DataContainer 的支持，并针对 KITTI 数据提供
专用的 collate_kitti（按字段语义拼接/填充/堆叠，构造 batch 字典）。

主要函数：
    - collate: 通用 collate，支持 DataContainer 的三种堆叠模式。
    - collate_kitti: 面向 KITTI 数据的专用 collate。
"""

import collections
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """把 batch 中每个字段整理为带 batch 维的张量或 DataContainer。

    在 default_collate 基础上支持 :type:`DataContainer`，对应三种情形：
    1. cpu_only = True：如元数据，按样本分组打包但不堆叠；
    2. cpu_only = False 且 stack = True：如图像张量，按样本堆叠；
    3. cpu_only = False 且 stack = False：如 GT 框列表，仅按组打包不堆叠。

    Args:
        batch: 待整理的样本序列。
        samples_per_gpu (int): 每张 GPU 的样本数，决定分组粒度。
    """
    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            # cpu_only：仅把样本原始数据按组放进列表，不搬 GPU、不堆叠
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True
            )
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    # 需要填充：统计组内最后 pad_dims 维的最大尺寸，
                    # 对不足者按 padding_value 补齐后再堆叠
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i : i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(
                                max_shape[dim - 1], sample.size(-dim)
                            )
                    padded_samples = []
                    for sample in batch[i : i + samples_per_gpu]:
                        # pad 列表按 (left,right) 成对组织，仅在最后 pad_dims 维做右填充
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value)
                        )
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    # 无需填充：直接按 default_collate 堆叠
                    stacked.append(
                        default_collate(
                            [sample.data for sample in batch[i : i + samples_per_gpu]]
                        )
                    )
                else:
                    raise ValueError("pad_dims should be either None or integers (1-3)")

        else:
            # stack=False：只按组打包、不堆叠
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i : i + samples_per_gpu]]
                )
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        # 序列字段逐列转置后递归 collate
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        # 映射字段逐键递归 collate
        return {
            key: collate([d[key] for d in batch], samples_per_gpu) for key in batch[0]
        }
    else:
        return default_collate(batch)



def collate_kitti(batch_list, samples_per_gpu=1):
    """面向 KITTI 数据的专用 collate。

    把一批样本按字段名归并，再按字段语义（拼接、填充、堆叠、坐标加 batch 偏移）构造
    最终 batch 字典。samples_per_gpu 参数当前未使用，仅保留接口一致性。

    Args:
        batch_list: 样本列表（元素为字典或字典列表）。
        samples_per_gpu (int): 每卡样本数（未使用）。

    Returns:
        dict: 整理完成的 batch 字典。
    """
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if type(example) is list:
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)
    batch_size = len(example_merged['metadata'])
    ret = {}
    # voxel_nums_list = example_merged["num_voxels"]
    # example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels",
                   "cyv_voxels", "cyv_num_points", "cyv_num_voxels"]:
            # 体素类数据沿轴 0 拼接
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in [
            "gt_boxes",
        ]:
            # GT 框按任务分别填充到 batch 内最大数量，再堆叠为 (batch, max_gt, 7)
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key == "metadata":
            # 元数据保持为原始列表
            ret[key] = elems
        elif key == "calib":
            # 标定信息按字段堆叠为 (batch, ...) 张量
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key == "points":
            # 原始点云逐样本保留为张量列表
            ret[key] = [torch.tensor(elem) for elem in elems]
        elif key in ["coordinates", "cyv_coordinates"]:
            # 坐标在列首插入 batch 内样本索引，用于区分体素归属，再沿轴 0 拼接
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0))
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                    "ind", "mask", "cat"]:
            # 这些字段为按 task 组织的列表，按 task 索引分别堆叠
            ret[key] = defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res
        elif key == 'gt_boxes_and_cls':
            # 直接堆叠为 (batch, ...) 张量
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
