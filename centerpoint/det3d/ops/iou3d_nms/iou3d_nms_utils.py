"""3D IoU 计算与旋转 NMS 的 Python 封装。

本模块是对 CUDA 扩展 iou3d_nms_cuda 的封装：负责把本仓库的旋转 3D 框格式
（[x, y, z, dx, dy, dz, heading]）转换为 pcdet 内核所需的坐标系后，调用 CUDA
内核完成 BEV/3D IoU 计算与 NMS。

主要函数：
    - boxes_iou_bev: 计算两组框在 BEV 平面的 IoU。
    - boxes_iou3d_gpu: 计算两组框的 3D IoU。
    - nms_gpu / nms_normal_gpu: 旋转框非极大值抑制。
"""

import torch

from . import iou3d_nms_cuda
import numpy as np 



def boxes_iou_bev(boxes_a, boxes_b):
    """计算两组旋转框在 BEV 平面上的 IoU。

    Args:
        boxes_a: (N, 7) 框 [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) 框 [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M) 两两框之间的 BEV IoU。
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou

def to_pcdet(boxes):
    """将本仓库的框格式转换为 pcdet 内核所需的坐标系。

    将尺寸顺序 (dx, dy, dz) 重排为 (dx, dz, dy)，并把 heading 取反再偏转
    -pi/2，以适配 pcdet 的角度约定。

    Args:
        boxes: (N, 7) 框 [x, y, z, dx, dy, dz, heading]

    Returns:
        boxes: (N, 7) 转换后的框。
    """
    # 转换回 pcdet 的坐标系约定。
    boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]
    boxes[:, -1] = -boxes[:, -1] - np.pi/2
    return boxes

def boxes_iou3d_gpu(boxes_a, boxes_b):
    """计算两组旋转框的 3D IoU（GPU 实现）。

    Args:
        boxes_a: (N, 7) 框 [x, y, z, dx, dy, dz, heading]
        boxes_b: (M, 7) 框 [x, y, z, dx, dy, dz, heading]

    Returns:
        iou3d: (N, M) 两两框之间的 3D IoU。
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # 先转换到 pcdet 坐标系
    boxes_a = to_pcdet(boxes_a)
    boxes_b = to_pcdet(boxes_b)

    # 高度方向重叠区间
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    # BEV 平面重叠面积
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3D 交集体积 = BEV 重叠面积 * 高度重叠
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """旋转框 NMS（GPU）。按分数降序，依次抑制 IoU 超过阈值的框。

    Args:
        boxes: (N, 7) 框 [x, y, z, dx, dy, dz, heading]
        scores: (N) 每个框的分数
        thresh: NMS 的 IoU 阈值
        pre_maxsize: 可选，最多保留分数最高的前 pre_maxsize 个框参与 NMS。

    Returns:
        保留框的索引 tensor 与 None（占位，保持调用接口一致）。
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """旋转框 NMS（GPU，普通版本）。实现与 nms_gpu 类似，调用 nms_normal_gpu 内核。

    Args:
        boxes: (N, 7) 框 [x, y, z, dx, dy, dz, heading]
        scores: (N) 每个框的分数
        thresh: NMS 的 IoU 阈值

    Returns:
        保留框的索引 tensor 与 None（占位，保持调用接口一致）。
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None