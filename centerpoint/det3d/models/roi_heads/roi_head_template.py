# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

"""RoI 头模板：目标分配、损失计算与预测框解码的公共实现。

对应论文 Sec. 3.4 的 refinement 训练监督：把第一阶段检测框作为 RoI，将 GT
变换到 RoI 的局部坐标系并编码为残差作为回归目标，分类目标则是 RoI 与 GT 的
IoU 置信度；损失由分类损失与 L1 回归损失组成。

主要类/函数：
    - limit_period: 角度周期折叠工具函数。
    - RoIHeadTemplate: 提供 assign_targets / get_loss / generate_predicted_boxes。

与其他模块关系：
    - 被 roi_head.py 的 RoIHead 继承复用；
    - 依赖 ProposalTargetLayer 完成 RoI 采样与标签生成。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.bbox import box_torch_ops
from .target_assigner.proposal_target_layer import ProposalTargetLayer

def limit_period(val, offset=0.5, period=np.pi):
    """把角度折叠到长度为 period 的周期区间内，避免朝向角超出合法范围。

    Args:
        val (Tensor): 输入角度（弧度）。
        offset (float): 平移量，决定区间起点；默认 0.5 使区间为 [-pi/2, 3pi/2)。
        period (float): 周期长度。

    Returns:
        Tensor: 折叠后的角度。
    """
    return val - torch.floor(val / period + offset) * period


class RoIHeadTemplate(nn.Module):
    """RoI 头的公共模板基类，负责目标分配、损失计算与预测框解码。

    具体网络结构由子类 RoIHead 定义，本类提供与结构无关的通用逻辑。
    """

    def __init__(self, num_class, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)

        self.forward_ret_dict = None

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        """构建由 1x1 卷积（全连接）+ BN + ReLU 堆叠的分支，末尾输出层带 bias。

        Args:
            input_channels (int): 输入通道数。
            output_channels (int): 输出通道数。
            fc_list (list[int]): 中间隐藏层通道列表。

        Returns:
            nn.Sequential: 构建好的分支网络。
        """
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            # kernel_size=1 的 Conv1d 等价于逐 RoI 的全连接层。
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            # 仅在第一层后按配置插入 Dropout，抑制过拟合。
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        # 最后一层输出分类 logits 或回归残差，带 bias 且无激活函数。
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    def assign_targets(self, batch_dict):
        """把 GT 变换到每个 RoI 的局部坐标系，构造回归/分类监督量。

        论文 Sec. 3.4：refinement 回归的是 RoI 到 GT 的残差，因此把 GT 平移到
        RoI 位置，并绕 z 轴旋转到与 RoI 朝向对齐的标准姿态，再编码成残差。

        Args:
            batch_dict (dict): 含 rois/gt_boxes_and_cls 等字段。

        Returns:
            dict: 写入 gt_of_rois（残差编码）等的 targets_dict。
        """
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            # 目标分配在 no_grad 下进行，不引入梯度。
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # yaw 折叠到 [-pi, pi]，避免朝向角因超过周期而造成旋转退化。
        roi_ry = limit_period(rois[:, :, 6], offset=0.5, period=np.pi*2)

        # 平移残差：GT 中心/尺寸/朝向减去 RoI 的对应量。
        gt_of_rois[:, :, :6] = gt_of_rois[:, :, :6] - rois[:, :, :6]
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # 绕 z 轴旋转 -roi_ry，把残差变换到 RoI 对齐的局部坐标系。
        gt_of_rois = box_torch_ops.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        if rois.shape[-1] == 9:
            # 含速度编码（x,y,z,w,l,h,yaw,vx,vy）：速度残差 = GT 速度 - RoI 速度。
            gt_of_rois[:, :, 7:-1] = gt_of_rois[:, :, 7:-1] - rois[:, :, 7:]    

            """
            roi_vel = gt_of_rois[:, :, 7:-1]
            roi_vel = torch.cat([roi_vel, torch.zeros([roi_vel.shape[0], roi_vel.shape[1], 1]).to(roi_vel)], dim=-1)

            gt_of_rois[:, :, 7:-1] = box_torch_ops.rotate_points_along_z(
                points=roi_vel.view(-1, 1, 3), angle=-roi_ry.view(-1)
            ).view(batch_size, -1, 3)[..., :2]
            """

        # 利用 3D box 的中心对称性消除相反朝向歧义：
        # 若残差朝向落在 (pi/2, 3pi/2) 区间，加 pi 后折叠，使相差 pi 的两个朝向
        # 回归到同一目标，再统一折叠与裁剪到 [-pi/2, pi/2]。
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label


        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        """计算 box 回归的 L1 损失，仅对前景 RoI（IoU 超阈值）生效。

        Args:
            forward_ret_dict (dict): 含 rcnn_reg/gt_of_rois/reg_valid_mask。

        Returns:
            (Tensor, dict): 回归损失与日志字典。
        """
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = forward_ret_dict['rcnn_reg'].shape[-1]
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        # 前景掩码：只监督 IoU 高于 REG_FG_THRESH 的 RoI。
        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'L1':
            reg_targets = gt_boxes3d_ct.view(rcnn_batch_size, -1)
            rcnn_loss_reg = F.l1_loss(
                rcnn_reg.view(rcnn_batch_size, -1),
                reg_targets,
                reduction='none'
            )  # [B, M, 7]

            # 各编码分量（中心/尺寸/朝向等）按配置权重加权。
            rcnn_loss_reg = rcnn_loss_reg * rcnn_loss_reg.new_tensor(\
                loss_cfgs.LOSS_WEIGHTS['code_weights'])

            # 仅累加前景样本的损失，并按前景数量归一化（无前景时除 1 防除零）。
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.detach()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        """计算分类（IoU 引导的置信度）损失，忽略标签为 -1 的中间样本。

        Args:
            forward_ret_dict (dict): 含 rcnn_cls/rcnn_cls_labels。

        Returns:
            (Tensor, dict): 分类损失与日志字典。
        """
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            # 标签 -1 表示处于前/背景阈值之间的中间样本，不参与损失。
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            # ignore_index=-1 让 CrossEntropy 自动忽略中间样本。
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.detach()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        """汇总分类与回归损失，返回第二阶段总损失与日志。

        Args:
            tb_dict (dict, optional): 已有日志字典，将被原地更新。

        Returns:
            (Tensor, dict): 总损失与更新后的日志字典。
        """
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """把局部残差解码回全局坐标下的精炼 box。

        解码是 assign_targets 编码的逆过程：残差先加上中心置零的 RoI 基准（保留
        尺寸与朝向），绕 z 轴旋转回原始朝向，再加回 RoI 中心得到全局坐标。

        Args:
            batch_size (int): 批大小。
            rois (Tensor): (B, N, 7+)，第一阶段 RoI。
            cls_preds (Tensor): (BN, num_class)，类别分数。
            box_preds (Tensor): (BN, code_size)，编码残差。

        Returns:
            (Tensor, Tensor): 类别分数 (B, N, num_class) 与精炼 box (B, N, code_size)。
        """
        code_size = box_preds.shape[-1]
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)

        # 局部 RoI 基准：中心置零、保留尺寸与朝向。
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        # 残差 + RoI 基准 = 局部精炼 box，随后绕 z 旋转回原始朝向。
        batch_box_preds = (batch_box_preds + local_rois).view(-1, code_size)
        batch_box_preds = box_torch_ops.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        # 加回 RoI 中心，恢复到全局坐标系。
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        
        return batch_cls_preds, batch_box_preds
