"""VoxelNet 检测器：基于体素化的单阶段检测器。

VoxelNet 继承 SingleStageDetector，其 extract_feat 支持两种输入形式：
(1) 原始点云列表，先调用 reader 做体素化；(2) 已体素化的特征，直接交给 reader
做体素特征编码。随后经 backbone（稀疏卷积）得到稠密 BEV 特征与多尺度稀疏体素
特征，再经 neck 与 bbox_head 输出检测结果。

对应论文 Sec 3：VoxelNet 是文中讨论的两种标准 3D backbone 之一。
"""

from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    """基于体素表示的单阶段检测器，继承自 SingleStageDetector。"""

    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        """构造 VoxelNet，直接复用父类组装 reader/backbone/neck/bbox_head。

        Args:
            reader (dict): 体素化 / 体素特征编码 reader 配置。
            backbone (dict): 稀疏卷积 backbone 配置。
            neck (dict): BEV neck 配置。
            bbox_head (dict): 检测头配置。
            train_cfg (dict, optional): 训练配置。
            test_cfg (dict, optional): 测试配置。
            pretrained (str, optional): 预训练权重路径。
        """
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        """提取体素特征并执行 backbone + neck。

        若数据中无 'voxels' 字段则输入为原始点云，先由 reader 完成体素化；
        否则使用已体素化的特征。

        Args:
            data: 输入数据 dict。

        Returns:
            Tuple[Tensor, dict]: (BEV 特征 x, backbone 的多尺度稀疏体素特征)。
        """
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            # 预体素化分支：直接用已分组好的体素特征做均值编码。
            input_features = self.reader(data["features"], data['num_voxels'])

        # backbone 接收 (体素特征, 体素坐标, batch 大小, 稀疏形状)，输出稠密 BEV 特征与多尺度稀疏体素特征。
        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        """单阶段前向：提取特征 → bbox_head 得到预测 → 计算损失或解码。

        Args:
            example: 输入数据。
            return_loss (bool): True 时计算损失，False 时返回解码后的预测结果。

        Returns:
            损失（训练）或预测结果（推理）。
        """
        x, _ = self.extract_feat(example)
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        """两阶段前向：返回一阶段预测框、BEV/体素/最终特征与一阶段损失。

        供 TwoStageDetector 调用，输出将交给第二阶段的 roi_head 做精炼。

        Args:
            example: 输入数据。
            return_loss (bool): True 时同时返回一阶段损失。

        Returns:
            推理返回 (boxes, bev_feature, voxel_feature, final_feat, None)；
            训练返回 (boxes, bev_feature, voxel_feature, final_feat, loss)。
        """
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # 手动深拷贝：detach 出无梯度的预测副本用于解码框，避免梯度穿过预测后处理。
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 