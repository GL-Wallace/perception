"""ResNet 2D backbone（备用）。

作为 torchie 训练框架提供的经典 2D 特征提取网络备选实现，支持 18/34/50/101/152
等深度，并可设置输出阶段、冻结层与 BN 冻结等选项。CenterPoint 主训练路径使用 3D
backbone，本文件保留作备用。支持从 checkpoint 路径加载预训练权重初始化。

主要类/函数：
    - conv3x3: 构造 3x3 卷积。
    - BasicBlock / Bottleneck: 两种残差块。
    - make_res_layer: 组装一层（stage）残差块。
    - ResNet: 完整 ResNet 结构。
"""

import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from ..trainer import load_checkpoint
from .weight_init import constant_init, kaiming_init


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    """ResNet 基础残差块，用于 ResNet-18/34。"""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 维度或分辨率不匹配时，用 1x1 卷积对齐捷径分支。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块，用于 ResNet-50/101/152。"""

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
    ):
        """瓶颈残差块。

        style 为 "pytorch" 时，步长为 2 的卷积是 3x3 卷积；为 "caffe" 时，
        步长为 2 的卷积是第一个 1x1 卷积。
        """
        super(Bottleneck, self).__init__()
        assert style in ["pytorch", "caffe"]
        if style == "pytorch":
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=conv1_stride, bias=False
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        def _inner_forward(x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out

        # 开启 with_cp 且需要梯度时，通过 checkpoint 用时间换显存。
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(
    block,
    inplanes,
    planes,
    blocks,
    stride=1,
    dilation=1,
    style="pytorch",
    with_cp=False,
):
    """组装一个残差 stage：首块负责下采样，其余块保持分辨率。"""
    downsample = None
    # 输入输出维度不一致或步长不为 1 时，需要 1x1 卷积对齐捷径分支。
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes, planes, stride, dilation, downsample, style=style, with_cp=with_cp
        )
    )
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes, planes, 1, dilation, style=style, with_cp=with_cp)
        )

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone。

    Args:
        depth (int): ResNet 深度，取值来自 {18, 34, 50, 101, 152}。
        num_stages (int): 残差 stage 数量，通常为 4。
        strides (Sequence[int]): 每个 stage 第一个块的步长。
        dilations (Sequence[int]): 每个 stage 的膨胀率。
        out_indices (Sequence[int]): 从哪些 stage 输出特征。
        style (str): `pytorch` 或 `caffe`。设为 "pytorch" 时步长为 2 的层是
            3x3 卷积，否则是第一个 1x1 卷积。
        frozen_stages (int): 需要冻结的 stage 数（所有参数固定）。-1 表示不冻结。
        bn_eval (bool): 是否将 BN 层设为 eval 模式，即冻结运行统计量（均值和方差）。
        bn_frozen (bool): 是否冻结 BN 层的权重和偏置。
        with_cp (bool): 是否使用 checkpoint。开启可节省显存，但会降低训练速度。
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        frozen_stages=-1,
        bn_eval=True,
        bn_frozen=False,
        with_cp=False,
    ):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError("invalid depth {} for resnet".format(depth))
        assert num_stages >= 1 and num_stages <= 4
        block, stage_blocks = self.arch_settings[depth]
        # 仅保留前 num_stages 个 stage。
        stage_blocks = stage_blocks[:num_stages]
        assert len(strides) == len(dilations) == num_stages
        assert max(out_indices) < num_stages

        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.with_cp = with_cp

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
            )
            self.inplanes = planes * block.expansion
            layer_name = "layer{}".format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = block.expansion * 64 * 2 ** (len(stage_blocks) - 1)

    def init_weights(self, pretrained=None):
        """初始化权重：可加载预训练 checkpoint，否则用 kaiming/常量初始化。"""
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            # 仅收集 out_indices 指定的 stage 输出。
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        """切换训练/评估模式，并按需冻结 BN 与前若干 stage 的参数。"""
        super(ResNet, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        if mode and self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            self.bn1.eval()
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for i in range(1, self.frozen_stages + 1):
                mod = getattr(self, "layer{}".format(i))
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False
