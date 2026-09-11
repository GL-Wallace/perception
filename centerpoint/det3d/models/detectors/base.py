"""CenterPoint 检测器的抽象基类定义。

本模块定义所有检测器（单阶段 / 两阶段）的公共基类 BaseDetector，声明了
reader / neck / bbox_head 等子模块的存在性判断属性，以及特征提取、训练与
测试推理的抽象接口，供具体检测器子类实现。

主要类：
    - BaseDetector: 所有检测器的抽象基类，继承自 nn.Module。

被 single_stage.py、two_stage.py 中的具体检测器类继承。
"""

import logging
from abc import ABCMeta, abstractmethod

import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn
from det3d import torchie


class BaseDetector(nn.Module):
    """所有检测器的抽象基类。

    提供 reader / neck / shared_head / bbox_head / mask_head 等子模块的
    存在性判定属性，并声明特征提取（extract_feat）与训练、推理的抽象接口，
    由具体检测器类（SingleStageDetector、TwoStageDetector）实现。
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_reader(self):
        """是否包含输入特征提取器（reader），负责对原始点云做体素/柱编码。"""
        return hasattr(self, "reader") and self.reader is not None

    @property
    def with_neck(self):
        """是否包含 neck，负责将 backbone 输出转为 BEV 特征。"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        """是否包含共享头（shared_head）。"""
        return hasattr(self, "shared_head") and self.shared_head is not None

    @property
    def with_bbox(self):
        """是否包含检测头（bbox_head），输出中心热图与回归量。"""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_mask(self):
        """是否包含分割头（mask_head）。"""
        return hasattr(self, "mask_head") and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """提取单帧输入的特征，由子类实现具体的 reader→backbone→neck 流程。

        Args:
            imgs: 单帧输入数据（点云及相关元信息）。

        Returns:
            提取到的特征张量。
        """
        pass

    def extract_feats(self, imgs):
        """对多帧输入逐帧提取特征并以生成器惰性返回。

        Args:
            imgs (list): 多帧输入列表。

        Yields:
            每帧经 extract_feat 提取到的特征。
        """
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, **kwargs):
        """训练阶段前向并计算损失，由子类实现。

        Args:
            imgs: 输入数据。
            **kwargs: 额外关键字参数。

        Returns:
            损失字典。
        """
        pass

    @abstractmethod
    def simple_test(self, img, **kwargs):
        """单尺度测试推理，由子类实现。

        Args:
            img: 输入数据。
            **kwargs: 额外关键字参数。

        Returns:
            预测结果。
        """
        pass

    @abstractmethod
    def aug_test(self, imgs, **kwargs):
        """多尺度增强（TTA）测试推理，由子类实现。

        Args:
            imgs: 多帧 / 多增广输入。
            **kwargs: 额外关键字参数。

        Returns:
            融合后的预测结果。
        """
        pass

    def init_weights(self, pretrained=None):
        """初始化模型权重，基类仅记录预训练权重路径。

        Args:
            pretrained (str, optional): 预训练权重路径，为 None 时不加载。
        """
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info("load model from: {}".format(pretrained))

    def forward_test(self, imgs, **kwargs):
        """测试阶段前向入口，由子类覆盖实现。"""
        pass

    def forward(self, example, return_loss=True, **kwargs):
        """前向入口，根据 return_loss 分发到训练或推理分支。

        Args:
            example: 输入数据。
            return_loss (bool): 是否计算损失（训练模式）。
            **kwargs: 额外关键字参数。
        """
        pass
