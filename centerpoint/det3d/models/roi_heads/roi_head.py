# ------------------------------------------------------------------------------
# Portions of this code are from
# OpenPCDet (https://github.com/open-mmlab/OpenPCDet)
# Licensed under the Apache License.
# ------------------------------------------------------------------------------

"""CenterPoint 两阶段 refinement 的第二阶段 RoI 头。

对应论文 Sec. 3.4：以第一阶段检测框采样得到的 RoI 特征为输入，先用共享的
1x1 卷积（等价于全连接）网络抽取特征，再分两路分别预测类置信度分数与 box
回归精炼量，恢复因下采样 stride 与有限感受野丢失的局部几何信息。

主要类：
    - RoIHead: 定义共享层与分类/回归两个输出分支。

与其他模块关系：
    - 由 det3d/models/detectors/two_stage.py 的 TwoStageDetector 实例化并调用；
    - 目标分配、损失计算与预测框解码复用 RoIHeadTemplate 的公共实现。
"""

from torch import batch_norm
import torch.nn as nn
import torch 
from .roi_head_template import RoIHeadTemplate

from det3d.core import box_torch_ops

from ..registry import ROI_HEAD

@ROI_HEAD.register_module
class RoIHead(RoIHeadTemplate):
    """第二阶段精炼头，对每个 RoI 输出分数与 box 残差（论文 Sec. 3.4 的 refine 分支）。"""

    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, add_box_param=False, test_cfg=None):
        """初始化共享全连接层与分类/回归分支。

        Args:
            input_channels (int): RoI 特征的通道数。
            model_cfg (config): 含 SHARED_FC/CLS_FC/REG_FC/DP_RATIO 等配置。
            num_class (int): 类别数。
            code_size (int): box 编码维度（7 或 9，9 表示额外含速度）。
            add_box_param (bool): 是否把 RoI 参数与分数拼到特征上一起回归。
            test_cfg (dict, optional): 推理配置，仅测试时使用。
        """
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg 
        self.code_size = code_size
        self.add_box_param = add_box_param

        pre_channel = input_channels

        # 共享层：1x1 卷积（等价于全连接）+ BN + ReLU 逐级堆叠；
        # 除最后一层外，按 DP_RATIO 在层间插入 Dropout 以缓解过拟合。
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        # 分类分支预测 IoU 引导的置信度分数，回归分支预测 box 编码残差。
        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        """初始化卷积权重；回归层最后一层用更小的标准差以稳定残差预测。

        Args:
            weight_init (str): 'kaiming' / 'xavier' / 'normal' 之一。
        """
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def forward(self, batch_dict, training=True):
        """第二阶段前向：RoI 特征 -> 共享层 -> 分类/回归；训练时先做目标分配。

        Args:
            batch_dict (dict): 含 rois/roi_scores/roi_labels/roi_features 等字段。
            training (bool): 训练或推理模式。

        Returns:
            dict: 训练时写入 rcnn_cls/rcnn_reg 供 get_loss 使用；
                  推理时写入 batch_cls_preds/batch_box_preds 供后处理解码。
        """
        batch_dict['batch_size'] = len(batch_dict['rois'])
        if training:
            # 为目标 RoI 分配 GT，得到编码残差、有效掩码与 IoU 等监督量。
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_features'] = targets_dict['roi_features']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        # RoI 特征准备：训练用采样后的特征，推理直接用第一阶段 RoI 特征。
        # 可选地把 box 参数与第一阶段分数拼入特征，让网络感知几何先验（RoI aware pooling）。
        if self.add_box_param:
            batch_dict['roi_features'] = torch.cat([batch_dict['roi_features'], batch_dict['rois'], batch_dict['roi_scores'].unsqueeze(-1)], dim=-1)

        pooled_features = batch_dict['roi_features'].reshape(-1, 1,
            batch_dict['roi_features'].shape[-1]).contiguous()  # (BxN, 1, C)

        batch_size_rcnn = pooled_features.shape[0]
        # 转置为 (BxN, C, 1)，把每个 RoI 作为一个长度为 1 的序列交给 Conv1d 处理。
        pooled_features = pooled_features.permute(0, 2, 1).contiguous() # (BxN, C, 1)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not training:
            # 推理：把编码残差解码回全局坐标，输出类别分数与精炼后的 box。
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            # 训练：保存分类/回归输出供 get_loss 计算损失。
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict
        
        return batch_dict        