"""2D CNN 子包。

汇总并对外导出备用的 2D backbone（ResNet/VGG/AlexNet）及其构造工具与权重初始化
函数。这些网络不参与 CenterPoint 的 3D 点云主训练路径，仅作为可选备份保留。
"""

from .alexnet import AlexNet
from .resnet import ResNet, make_res_layer
from .vgg import VGG, make_vgg_layer
from .weight_init import (
    caffe2_xavier_init,
    constant_init,
    kaiming_init,
    normal_init,
    uniform_init,
    xavier_init,
)

__all__ = [
    "AlexNet",
    "VGG",
    "make_vgg_layer",
    "ResNet",
    "make_res_layer",
    "constant_init",
    "xavier_init",
    "normal_init",
    "uniform_init",
    "kaiming_init",
    "caffe2_xavier_init",
]
