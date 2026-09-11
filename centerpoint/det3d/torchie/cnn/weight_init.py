"""权重初始化工具。

对 PyTorch 的各类权重初始化函数做薄封装，统一了「初始化权重 + 可选地将偏置置零」
的调用方式，供 cnn 子包中的 backbone 在 init_weights 阶段复用（如 ResNet/VGG 的
kaiming_init 与 BN 的 constant_init）。

主要函数：
    - constant_init / normal_init / uniform_init: 常量、正态、均匀分布初始化。
    - xavier_init / kaiming_init: 针对卷积网络的两种自适应初始化策略。
    - caffe2_xavier_init: 对齐 Caffe2 XavierFill 行为的便利封装。
"""

import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    """Kaiming（He）初始化，适配 ReLU 类激活函数以保持激活方差稳定。

    各参数含义与 PyTorch 的 kaiming_uniform_/kaiming_normal_ 一致。
    """
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # Caffe2 的 XavierFill 等价于 PyTorch 中以 leaky_relu、fan_in 模式进行的
    # kaiming_uniform_ 初始化（此结论来自 FAIR 内部代码）。
    kaiming_init(
        module, a=1, mode="fan_in", nonlinearity="leaky_relu", distribution="uniform"
    )
