"""数值类型与形状检查工具。"""

import numpy as np


def is_array_like(x):
    """判断 x 是否为类数组对象（list/tuple/ndarray）。"""
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    """判断 x 的形状是否与期望形状可合并。

    仅当二者均为类数组且维度一致时，逐维检查：期望形状中非 None 的维度必须与
    实际维度相等。

    Args:
        x: 待检查的值。
        expected_shape: 期望形状（其中 None 表示该维度不参与约束）。

    Returns:
        bool: 可合并返回 True，否则 False。
    """
    mergeable = True
    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable
