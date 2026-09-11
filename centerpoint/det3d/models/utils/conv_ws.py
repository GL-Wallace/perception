"""带权重标准化（Weight Standardization）的 2D 卷积。

在每次前向时对卷积核逐输出通道做标准化（减均值、除标准差），再做标准卷积，
有助于平滑损失景观、提升训练稳定性。

主要函数 / 类：
    - conv_ws_2d: 权重标准化的 2D 卷积前向。
    - ConvWS2d: 继承 nn.Conv2d 的权重标准化卷积层。
"""

import torch.nn as nn
import torch.nn.functional as F


def conv_ws_2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-5
):
    """对卷积核做权重标准化后执行 2D 卷积。

    对每个输出通道的卷积核元素做 z-score 标准化（减均值、除标准差），其中
    标准差按有偏估计计算（除以 N 而非 N-1）。

    Args:
        input (Tensor): 卷积输入。
        weight (Tensor): 卷积核权重。
        bias (Tensor, optional): 偏置。
        stride / padding / dilation / groups: 与 F.conv2d 同义。
        eps (float): 防止除零的极小值。

    Returns:
        Tensor: 卷积输出。
    """
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    # 逐输出通道计算权重均值与标准差。
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):
    """权重标准化的 2D 卷积层，参数与 nn.Conv2d 一致。"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        eps=1e-5,
    ):
        """构造权重标准化卷积层。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size: 卷积核尺寸。
            stride / padding / dilation / groups / bias: 与 nn.Conv2d 同义。
            eps (float): 权重标准化的防除零极小值。
        """
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        """执行权重标准化卷积前向。"""
        return conv_ws_2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.eps,
        )
