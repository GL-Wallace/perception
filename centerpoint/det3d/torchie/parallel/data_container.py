"""自定义数据容器 DataContainer。

为 dataloader 的 collate/scatter 提供可携带任意类型数据的容器，解决“所有张量
必须同尺寸、类型受限”的限制。支持 stack 拼接、GPU 拷贝、padding 与保留原样四种
行为，配合 MMDataParallel 在分布式数据并行中搬运 batch。
"""
import functools

import torch


def assert_tensor_type(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                "{} has no attribute {} for type {}".format(
                    args[0].__class__.__name__, func.__name__, args[0].datatype
                )
            )
        return func(*args, **kwargs)

    return wrapper


class DataContainer(object):
    """可容纳任意类型对象的数据容器。

    通常张量会在 collate 阶段被拼接、在 scatter 阶段沿某维切分，但这种行为有局限：
    1. 所有张量必须相同尺寸；
    2. 类型受限（仅 numpy 数组或 Tensor）。

    DataContainer 与 MMDataParallel 用于突破这些限制，支持以下行为：
        - 拷贝到 GPU：把所有张量 pad 到相同尺寸后 stack；
        - 拷贝到 GPU 但不 stack；
        - 原样保留对象并直接传给模型；
        - 通过 pad_dims 指定最后几维进行 padding。
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()
