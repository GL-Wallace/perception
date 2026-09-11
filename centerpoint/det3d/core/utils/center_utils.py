# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou and Tianwei Yin 
# ------------------------------------------------------------------------------
"""CenterPoint 的工具函数集合，主要服务于中心热图的生成与索引 gather。

包含热图相关（gaussian_radius/gaussian2D/draw_umich_gaussian）、特征 gather
（_gather_feat/_transpose_and_gather_feat）以及圆形 NMS/双线性插值等辅助函数。

主要函数：
    - gaussian_radius: 依据物体 BEV 尺寸与 min_overlap 计算高斯核半径（论文 Sec 3.1）。
    - gaussian2D: 生成 2D 高斯核。
    - draw_umich_gaussian: 把 2D 高斯放置到热图的物体中心处（论文 Sec 3.1）。
    - _gather_feat / _transpose_and_gather_feat: 在展平索引 ind 处 gather 特征。
    - _circle_nms: 基于中心点距离的圆形 NMS。
    - bilinear_interpolate_torch: 双线性插值。

设计思路：
    中心热图的 GT 由高斯核在中心处软化得到，半径与物体 BEV 尺寸及允许的重叠程度相关，
    使相邻物体中心尽量可区分。gather 函数将 CHW 特征转成 (B, H*W, C) 后按展平索引
    ind = y * W + x 取出正样本预测，供回归损失与 focal loss 使用。

与其他模块的关系：
    - gaussian_radius/draw_umich_gaussian 被 det3d.datasets.pipelines.preprocess.AssignLabel 调用，
      用于构造训练时的热图监督。
    - _transpose_and_gather_feat 被 det3d.models.losses.centernet_loss 的 RegLoss/FastFocalLoss
      调用。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from .circle_nms_jit import circle_nms

def gaussian_radius(det_size, min_overlap=0.5):
    """依据目标 BEV 上的尺寸推导 2D 高斯核半径（论文 Sec 3.1）。

    从 CornerNet 继承：分别以三种几何约束（两框内切、外接、允许重叠）建立一元二次方程，
    求解使两个相邻框的 IoU 恰好达到 min_overlap 的半径，取三者最小值，
    保证相邻中心的高斯峰足够独立、尽量不互相淹没。

    Args:
        det_size (Tuple[float, float]): 目标在 BEV 平面上的高与宽（单位：feature 格）。
        min_overlap (float): 相邻两框允许的最小交并比，用于推导半径。

    Returns:
        float: 高斯核半径。物体越大半径越大，用于绘制平滑的中心置信区域。
    """
    height, width = det_size

    # 约束一：两框相交区域近似外接正方形，半径使相交面积满足 overlap 阈值。
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    # 约束二：两框其中一个完全包含另一视角下边界时的解。
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    # 约束三：两框允许相互重叠 min_overlap 时的解。
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    """生成中心峰值为 1 的 2D 高斯核。

    Args:
        shape (Tuple[int, int]): 高斯核高与宽。
        sigma (float): 高斯标准差。

    Returns:
        np.ndarray: 2D 高斯核，峰值附近小于极小值的元素置 0。
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # 将远小于峰值的数值置 0，得到稀疏高斯核，加速后续逐像素取 max 操作。
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    """把 2D 高斯核画到热图的物体中心处（论文 Sec 3.1 的软化监督）。

    以 (center) 为中心、radius 为半径，将高斯核与热图逐位置取最大写入，
    使重叠物体的多个高斯峰共享热图时保留更高置信度。

    Args:
        heatmap (np.ndarray): 待绘制的热图（单通道，shape (H, W)）。
        center (np.ndarray): 物体中心坐标 (x, y)。
        radius (int): 高斯核半径。
        k (float): 高斯峰值缩放系数，默认 1 表示峰值处置信度为 1。

    Returns:
        np.ndarray: 原地更新后的热图。
    """
    diameter = 2 * radius + 1
    # sigma 取 diameter/6，与 CornerNet 一致，使高斯峰覆盖绝大多数相关区域。
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # 计算高斯与热图的有效重叠区域，处理中心贴近边界的越界情况。
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _gather_feat(feat, ind, mask=None):
    """在维度 1 上按索引 ind 收集特征。

    Args:
        feat (Tensor): 特征，shape 为 (batch, N, dim)。
        ind (Tensor): 索引，shape 为 (batch, M)。
        mask (Tensor | None): 若给定，收集后再按 mask 过滤有效位置。

    Returns:
        Tensor: gather 后的特征，shape 为 (batch, M, dim)（有 mask 时展平为 (-1, dim)）。
    """
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    """把 CHW 特征转成 (B, H*W, C) 后按展平索引 ind 收集。

    论文 Sec 3.3 中回归损失只监督 GT 中心位置，这里通过 ind = y * W + x
    把 2D 中心坐标展平成 1D 索引，从而在宽高合并后的维度上 gather 出正样本预测。

    Args:
        feat (Tensor): 特征，shape 为 (B, C, H, W)。
        ind (Tensor): 展平索引，shape 为 (B, M)。

    Returns:
        Tensor: 收集后的特征，shape 为 (B, M, C)。
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _circle_nms(boxes, min_radius, post_max_size=83):
    """按中心点间距离做 NMS（圆形 NMS）。

    Args:
        boxes (Tensor): (N, 3)，前两列为中心 x/y，第三列为置信度。
        min_radius (float): 中心最小间距阈值。
        post_max_size (int): 最多保留的框数量。

    Returns:
        Tensor: 保留框的下标。
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep 


def bilinear_interpolate_torch(im, x, y):
    """对输入图像在给定坐标 (x, y) 处做双线性插值采样。

    Args:
        im: 特征图，shape 为 (H, W, C)，按 [y, x] 索引。
        x: 查询点横坐标，shape 为 (N)。
        y: 查询点纵坐标，shape 为 (N)。

    Returns:
        Tensor: 插值后的特征，shape 为 (C, N)。
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    # 将采样点邻域钳制到图像范围内，避免越界。
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    # 依据查询点与四个邻域像素的距离计算双线性权重并加权求和。
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans
