"""网络构建的通用辅助工具。

提供自定义 Sequential 容器、GroupNorm / Empty 占位模块，以及 inspect 辅助函数、
默认参数修改装饰器、梯度打印钩子与 padding 指示掩码生成等工具。

主要函数 / 类：
    - Sequential / GroupNorm / Empty: 网络模块。
    - get_pos_to_kw_map / get_kw_to_default_map: 函数签名辅助函数。
    - change_default_args: 修改层默认参数的装饰器工厂。
    - get_printer / register_hook: 梯度调试钩子。
    - get_paddings_indicator: 生成 padding 掩码。
"""

import functools
import inspect
import sys
from collections import OrderedDict

import numba
import numpy as np
import torch

# from lib.models.backbone.utils import Registry
#
# BACKBONES = Registry()
# RPN_HEADS = Registry()
# ROI_BOX_FEATURE_EXTRACTORS = Registry()
# ROI_BOX_PREDICTOR = Registry()
# ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
# ROI_KEYPOINT_PREDICTOR = Registry()
# ROI_MASK_FEATURE_EXTRACTORS = Registry()
# ROI_MASK_PREDICTOR = Registry()


class Sequential(torch.nn.Module):
    """自定义顺序容器，功能类似 nn.Sequential。

    按传入顺序（位置参数、OrderedDict 或关键字参数）依次添加子模块，并提供按
    整数下标的访问（__getitem__）与运行时追加（add）能力。
    """

    def __init__(self, *args, **kwargs):
        """构造顺序容器，支持位置参数、OrderedDict 与关键字参数三种传入方式。"""
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        """按整数下标访问子模块，支持负索引。"""
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        """追加一个子模块，名称缺省时使用当前长度作为键。"""
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        """依序把输入传入每个子模块。"""
        # i = 0
        for module in self._modules.values():
            # print(i)
            input = module(input)
            # i += 1
        return input


class GroupNorm(torch.nn.GroupNorm):
    """GroupNorm 封装，接口与 torch.nn.GroupNorm 一致。"""

    def __init__(self, num_channels, num_groups, eps=1e-5, affine=True):
        """构造 GroupNorm。

        Args:
            num_channels (int): 通道数。
            num_groups (int): 分组数。
            eps (float): 数值稳定项。
            affine (bool): 是否使用可学习仿射参数。
        """
        super().__init__(
            num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine
        )


class Empty(torch.nn.Module):
    """占位模块：不改变输入，直接透传。"""

    def __init__(self, *args, **kwargs):
        """占位构造，忽略所有参数。"""
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        """透传输入：单个参数返回自身，无参数返回 None，否则返回参数元组。"""
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def get_pos_to_kw_map(func):
    """返回函数「位置序号 → 参数名」的映射（仅限 POSITIONAL_OR_KEYWORD 参数）。

    Args:
        func: 目标函数。

    Returns:
        dict: 位置序号到参数名的映射。
    """
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_kw_to_default_map(func):
    """返回函数「参数名 → 默认值」的映射（仅含拥有默认值的可位置/关键字参数）。"""
    kw_to_default = {}
    fsig = inspect.signature(func)
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            if info.default is not info.empty:
                kw_to_default[name] = info.default
    return kw_to_default


def change_default_args(**kwargs):
    """返回一个装饰器，用于给目标层类的构造参数设置默认值。

    当调用方未显式传入被覆盖的关键字参数时，使用这里给定的默认值。
    """

    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper


def get_printer(msg):
    """返回一个打印函数，用于在反向传播的 hook 中打印张量或其梯度统计。

    Args:
        msg (str): 打印前缀。

    Returns:
        callable: 打印函数。
    """

    def printer(tensor):
        if tensor.nelement() == 1:
            print(f"{msg} {tensor}")
        else:
            print(
                f"{msg} shape: {tensor.shape}"
                f" max: {tensor.max()} min: {tensor.min()}"
                f" mean: {tensor.mean()}"
            )

    return printer


def register_hook(tensor, msg):
    """对张量同时调用 retain_grad 与 register_hook，用于打印梯度。"""
    tensor.retain_grad()
    tensor.register_hook(get_printer(msg))


def get_paddings_indicator(actual_num, max_num, axis=0):
    """生成表示「真实位置」的布尔掩码（用于带 padding 的张量）。

    Args:
        actual_num (Tensor): 每条记录的真实长度（如每柱的真实点数）。
        max_num (int): padding 后的最大长度。
        axis (int): actual_num 被 unsqueeze 的维度（表示沿哪个轴作比较）。

    Returns:
        Tensor: 布尔掩码，真实位置为 True。
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # 构造从 0 到 max_num-1 的下标，与真实长度比较得到掩码。
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator
