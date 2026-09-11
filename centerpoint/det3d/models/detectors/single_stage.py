"""单阶段检测器的组装实现。

SingleStageDetector 按 reader → backbone → neck → bbox_head 的顺序组装网络，是
CenterPoint 论文 Sec 3 标准骨架（点云编码 → 稀疏卷积 backbone → BEV neck →
CenterHead 检测头）的检测器壳子。VoxelNet 与 PointPillars 均继承自本类。

主要类：
    - SingleStageDetector: 负责子模块构建、特征提取与权重初始化。
"""

import torch.nn as nn

from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from ..utils.finetune_utils import FrozenBatchNorm2d
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class SingleStageDetector(BaseDetector):
    """单阶段检测器，将 reader/backbone/neck/bbox_head 组装为完整前向流程。

    对应论文 Sec 3 的骨架结构：raw point cloud → reader(体素/柱编码) →
    backbone(稀疏卷积) → neck(BEV 特征) → bbox_head(CenterHead)。
    """

    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        """通过 builder 按配置实例化 reader/backbone/neck/bbox_head 等子模块。

        Args:
            reader (dict): 输入特征提取器（reader）配置。
            backbone (dict): 稀疏卷积 backbone 配置。
            neck (dict, optional): BEV 特征转换 neck 配置，可为 None。
            bbox_head (dict, optional): 检测头配置，可为 None。
            train_cfg (dict, optional): 训练配置。
            test_cfg (dict, optional): 测试配置。
            pretrained (str, optional): 预训练权重路径。
        """
        super(SingleStageDetector, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """从预训练权重加载参数，失败时打印提示而不中断。

        Args:
            pretrained (str, optional): 预训练权重路径。
        """
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))
            
    def extract_feat(self, data):
        """执行 reader → backbone → neck 的特征提取流程。

        Args:
            data: 输入数据（含 features/coors/num_voxels 等字段）。

        Returns:
            经 neck 处理后的 BEV 特征张量（若不含 neck 则为 backbone 输出）。
        """
        input_features = self.reader(data)
        x = self.backbone(input_features)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, example, rescale=False):
        """多尺度（TTA）测试推理，单阶段检测器暂未实现。

        Args:
            example: 输入数据。
            rescale (bool): 是否将预测框缩放还原。
        """
        raise NotImplementedError

    def forward(self, example, return_loss=True, **kwargs):
        """前向入口，由具体子类（VoxelNet / PointPillars）覆盖实现。"""
        pass

    def predict(self, example, preds_dicts):
        """推理阶段将检测头输出解码为最终预测框，由子类覆盖实现。

        Args:
            example: 输入数据。
            preds_dicts: 检测头输出的原始预测。
        """
        pass

    def freeze(self):
        """冻结所有参数并将 BatchNorm 转为 FrozenBatchNorm2d。

        用于两阶段训练时冻结第一阶段网络。

        Returns:
            Self，便于链式调用。
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self