"""pipeline 组合器（Compose）。

Compose 把配置 dict 列表实例化为一系列可调用变换，并依次应用到每个样本上；
这是数据预处理流水线（加载 → 增强 → 体素化 → 标签分配 → 格式化）的执行入口。
"""
import collections

from det3d.utils import build_from_cfg
from ..registry import PIPELINES


@PIPELINES.register_module
class Compose(object):
    """按顺序执行一组数据变换。

    transforms 中的 dict 项通过 build_from_cfg 由注册表实例化（type 为 'Empty'
    时跳过），callable 项直接使用。
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Empty':
                    continue 
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, res, info):
        """依次执行每个变换，并在任一变换使 res 变为 None 时提前返回 None。

        Args:
            res (dict): 输入数据字典。
            info (dict): 样本 info。

        Returns:
            tuple 或 None: (res, info)。
        """
        for t in self.transforms:
            res, info = t(res, info)
            if res is None:
                return None
        return res, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

