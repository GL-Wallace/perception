"""卷积模块组装工具。

提供按配置构建卷积层的 build_conv_layer，以及包含 conv / norm / activation 的
通用 ConvModule，可按 order 灵活组合各子层顺序。

主要函数 / 类：
    - build_conv_layer: 按配置构建卷积层（Conv / ConvWS）。
    - ConvModule: conv + norm + activation 的通用卷积块。
"""

import warnings

import torch.nn as nn
from det3d.torchie.cnn import constant_init, kaiming_init

from .conv_ws import ConvWS2d
from .norm import build_norm_layer

conv_cfg = {
    "Conv": nn.Conv2d,
    "ConvWS": ConvWS2d,
    # TODO: octave conv
}


def build_conv_layer(cfg, *args, **kwargs):
    """按配置构建卷积层。

    Args:
        cfg (None or dict): 卷积配置，需含 type 字段；为 None 时默认 Conv。
        *args / **kwargs: 传入卷积层构造函数的参数。

    Returns:
        nn.Module: 创建的卷积层。
    """
    if cfg is None:
        cfg_ = dict(type="Conv")
    else:
        assert isinstance(cfg, dict) and "type" in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in conv_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    """包含 conv / norm / activation 的通用卷积块。

    按 order 指定的顺序组合卷积、归一化与激活，支持自动处理 bias（归一化前置时
    默认关闭卷积偏置）。

    Args:
        in_channels (int): 输入通道数（同 nn.Conv2d）。
        out_channels (int): 输出通道数。
        kernel_size: 卷积核尺寸。
        stride / padding / dilation / groups: 同 nn.Conv2d。
        bias (bool | str): 'auto' 时依据是否含 norm 自动决定。
        conv_cfg (dict): 卷积层配置。
        norm_cfg (dict): 归一化层配置。
        activation (str | None): 激活类型，仅支持 'relu'。
        inplace (bool): 是否原地执行激活。
        order (tuple[str]): conv / norm / act 的排列顺序。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        activation="relu",
        inplace=True,
        order=("conv", "norm", "act"),
    ):
        """构造卷积块并按 order 组装 conv / norm / act 子层。"""
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(["conv", "norm", "act"])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # 卷积后若紧跟归一化层，则卷积偏置冗余，默认关闭。
        if bias == "auto":
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn("ConvModule has norm and bias at the same time")

        # 构建卷积层。
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # 将卷积层的属性提升到本层，便于外部访问。
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # 构建归一化层。
        if self.with_norm:
            # 归一化层位于卷积之后时，其通道数取输出通道，否则取输入通道。
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # 构建激活层。
        if self.with_activatation:
            # TODO: 后续可引入 act_cfg 支持更多激活类型。
            if self.activation not in ["relu"]:
                raise ValueError(
                    "{} is currently not supported.".format(self.activation)
                )
            if self.activation == "relu":
                self.activate = nn.ReLU(inplace=inplace)

        # 默认使用 msra（kaiming）初始化。
        self.init_weights()

    @property
    def norm(self):
        """返回归一化层。"""
        return getattr(self, self.norm_name)

    def init_weights(self):
        """对卷积层用 kaiming 初始化，对归一化层做常量初始化。"""
        nonlinearity = "relu" if self.activation is None else self.activation
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        """按 order 依次执行 conv / norm / act。

        Args:
            x (Tensor): 输入。
            activate (bool): 是否执行激活层。
            norm (bool): 是否执行归一化层。

        Returns:
            Tensor: 输出特征。
        """
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activatation:
                x = self.activate(x)
        return x
