"""数据 scatter 工具。

把 batch 数据按目标 GPU 列表切分，相比 PyTorch 原版增加了对 DataContainer
的支持：cpu_only 的容器直接原样传给每个 GPU，其余容器按第 0 维切片。

主要函数：
    - scatter: 递归散列张量 / DataContainer / 容器结构到各 GPU。
    - scatter_kwargs: 同时散列位置参数与关键字参数。
"""

import torch
from torch.nn.parallel._functions import Scatter as OrigScatter

from ._functions import Scatter
from .data_container import DataContainer


def scatter(inputs, target_gpus, dim=0):
    """把输入散列到目标 GPU 列表。

    相比 :func:`torch.nn.parallel.scatter_gather.scatter`，本函数额外支持
    :class:`DataContainer`：cpu_only 容器直接返回原始数据（不搬 GPU），
    其余容器按 dim 维切片后分发。

    Args:
        inputs: 待散列的输入（张量、DataContainer 或嵌套结构）。
        target_gpus (list[int]): 目标 GPU 编号列表。
        dim (int): 切片维度。

    Returns:
        list: 每个目标 GPU 对应的一份输入。
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, DataContainer):
            if obj.cpu_only:
                # cpu_only 的数据（如元信息）不搬 GPU，各卡共享同一份
                return obj.data
            else:
                return Scatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        # 其他不可散列对象直接复制到每个目标 GPU
        return [obj for targets in target_gpus]

    # scatter_map 递归调用自身形成闭包引用；调用后置 None 以解除引用环
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """散列位置参数与关键字参数。

    Args:
        inputs: 位置参数。
        kwargs: 关键字参数字典。
        target_gpus (list[int]): 目标 GPU 列表。
        dim (int): 切片维度。

    Returns:
        tuple: (inputs 元组, kwargs 元组)，长度与 target_gpus 一致，
        并自动用空 tuple/空 dict 补齐较短的一方。
    """
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
