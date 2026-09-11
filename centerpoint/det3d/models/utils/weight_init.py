"""权重初始化工具函数集合。

封装 torch.nn.init 的常用初始化方法，并统一处理偏置的常量初始化，供网络模块在
构造时调用。

主要函数：
    - xavier_init / normal_init / uniform_init / kaiming_init: 各类权重初始化。
    - bias_init_with_prob: 依据先验概率计算偏置初始值（如 CenterHead 热图分支）。
"""

import numpy as np
import torch.nn as nn


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    """对模块权重做 Xavier 初始化。

    Args:
        module (nn.Module): 含 weight 的目标模块。
        gain (float): 增益系数。
        bias (float): 偏置初始值。
        distribution (str): 'uniform' 或 'normal'。
    """
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    """对模块权重做正态分布初始化。

    Args:
        module (nn.Module): 含 weight 的目标模块。
        mean (float): 均值。
        std (float): 标准差。
        bias (float): 偏置初始值。
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    """对模块权重做均匀分布初始化。

    Args:
        module (nn.Module): 含 weight 的目标模块。
        a (float): 区间下界。
        b (float): 区间上界。
        bias (float): 偏置初始值。
    """
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    """对模块权重做 Kaiming（He）初始化。

    Args:
        module (nn.Module): 含 weight 的目标模块。
        mode (str): 'fan_in' 或 'fan_out'。
        nonlinearity (str): 激活函数类型（如 relu / leaky_relu）。
        bias (float): 偏置初始值。
        distribution (str): 'uniform' 或 'normal'。
    """
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias"):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """根据先验概率计算 conv/fc 偏置的初始值。

    使 sigmoid(初始 logits) 约等于给定先验概率，从而缓解类别不平衡（常用于
    CenterHead 热图分支的偏置初始化）。

    Args:
        prior_prob (float): 期望的初始正样本概率。

    Returns:
        float: 偏置初始值。
    """
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init
