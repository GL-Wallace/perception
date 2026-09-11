"""CenterPoint 的损失函数模块（论文 Sec 3.3）。

实现热图监督使用的 penalty-reduced Focal Loss 与仅在 GT 中心位置计算的
L1 回归损失。二者均基于「稀疏正样本 + 展平索引 gather」的方式实现，
以降低显存占用与计算量。

主要类：
    - RegLoss: 在 GT 中心(ind)位置 gather 预测值并计算 L1 回归损失。
    - FastFocalLoss: 复刻 CornerNet 的 focal loss，等价于论文的 penalty-reduced focal loss。

设计思路：
    热图是逐像素的 K 通道分类问题，GT 只在物体中心处为 1（其余被 2D 高斯软化），
    focal loss 用 (1-p)^2 降低简单负样本权重、用惩罚项 (1-Y)^4 降低靠近 GT 中心附近的
    负样本贡献（penalty-reduced）。回归损失只在正样本中心处求 L1，避免背景像素参与。

与其他模块的关系：
    - FastFocalLoss/RegLoss 由 det3d.models.bbox_heads.center_head 的 CenterHead 实例化并调用。
    - 二者依赖 det3d.core.utils.center_utils._transpose_and_gather_feat 在展平索引上 gather。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat

class RegLoss(nn.Module):
  """仅在 GT 中心位置计算的 L1 回归损失（论文 Sec 3.3）。

  注意:
    output 为 CHW 排列的预测图，通过 ind 展平索引 gather 出正样本预测，
    再用 mask 屏蔽无效位置，按正样本数量归一化后输出每个回归维度的损失。
  """
  def __init__(self):
    super(RegLoss, self).__init__()

  def forward(self, output, mask, ind, target):
    """计算回归损失。

    Args:
        output (Tensor): 预测图，shape 为 (batch, dim, h, w)。
        mask (Tensor): 有效目标掩码，shape 为 (batch, max_objects)，1 表示该位置存在 GT。
        ind (Tensor): GT 中心的展平索引（ind = y * W + x），shape 为 (batch, max_objects)。
        target (Tensor): 回归目标，shape 为 (batch, max_objects, dim)。

    Returns:
        Tensor: 每个 batch 的回归损失，shape 为 (dim, batch)，各回归维度独立统计。
    """
    # 将 CHW 转成 (batch, h*w, dim) 后在 ind 处 gather 正样本预测。
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    # 用 mask 屏蔽无效位置的预测与目标，仅在正样本中心计算 L1 损失。
    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class FastFocalLoss(nn.Module):
  """penalty-reduced focal loss（论文 Sec 3.3，公式(2)）。

  复刻 CornerNet 实现，相比逐像素 softmax 版本更快、显存占用更小：
  负样本损失直接在整张预测图上按公式计算，正样本仅通过 ind 索引 gather。

  注意:
    输入 out 需先经过 sigmoid（在 CenterHead.loss 中完成），这里直接取 log，
    因此 out 必须被截断到 (0, 1) 区间以避免数值问题。
  """
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, mask, cat):
    """计算热图 focal loss。

    Args:
        out (Tensor): sigmoid 后的预测热图，shape 为 (B, C, H, W)。
        target (Tensor): 热图 GT，shape 为 (B, C, H, W)。
        ind (Tensor): GT 中心展平索引，shape 为 (B, M)。
        mask (Tensor): 有效目标掩码，shape 为 (B, M)。
        cat (Tensor): 每个 GT 中心的类别 id，shape 为 (B, M)。

    Returns:
        Tensor: 标量 focal loss。若无正样本则退化为纯负样本损失。
    """
    mask = mask.float()
    # 论文公式(2) 中的惩罚项 (1 - Y_xyz)^alpha，此处 alpha=4：
    # 目标越接近 1 的正样本附近，负样本损失被越强地抑制（penalty-reduced）。
    gt = torch.pow(1 - target, 4)
    # 负样本 focal loss：-log(1-p) * p^2 * 惩罚项，直接在全图上求和。
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    # 在正样本位置按类别取出对应通道的预测置信度。
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    # 正样本 focal loss：-log(p) * (1-p)^2，用 mask 屏蔽无效位置。
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos