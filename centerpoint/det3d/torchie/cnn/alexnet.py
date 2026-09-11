"""AlexNet 2D backbone（备用）。

作为 torchie 训练框架提供的经典 2D 分类网络，仅作为备选 backbone 保留；CenterPoint
实际使用的是基于 lidar 点云的 3D backbone，该类不在主训练路径上。支持通过字符串
路径加载预训练 checkpoint 初始化权重。

主要类：
    - AlexNet: 经典 AlexNet 结构。
"""

import logging

import torch.nn as nn

from ..trainer import load_checkpoint


class AlexNet(nn.Module):
    """AlexNet backbone。

    Args:
        num_classes (int): 分类类别数。若小于等于 0 则不构建分类头，
            仅输出特征图。
    """

    def __init__(self, num_classes=-1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        # 特征提取部分：5 组卷积 + 池化。
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 仅当需要分类时构建全连接分类头。
        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def init_weights(self, pretrained=None):
        """初始化权重，可选择从 checkpoint 加载预训练参数。

        传入字符串路径时以非严格方式加载对应 checkpoint；传 None 时使用默认初始化。
        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            # use default initializer
            pass
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):

        x = self.features(x)
        if self.num_classes > 0:
            # 展平后送入全连接分类头。
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)

        return x
