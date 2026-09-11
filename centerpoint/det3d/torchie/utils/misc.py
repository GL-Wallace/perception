"""通用杂项工具。

提供类型判断、迭代器/列表转换、序列切片/拼接以及前置依赖检查等小型通用函数，
被 fileio、config、progressbar 等其他 torchie 模块广泛复用。

主要函数：
    - is_str / is_seq_of / is_list_of / is_tuple_of: 类型判断。
    - iter_cast / list_cast / tuple_cast: 迭代器元素类型转换。
    - slice_list / concat_list: 列表切片与拼接。
    - check_prerequisites / requires_package / requires_executable: 依赖检查装饰器。
"""

import collections
import functools
import itertools
import subprocess
from importlib import import_module

import six

# collections 的 ABC 在 python 3.8+ 将被弃用，而 collections.abc 在 python 2.7
# 中不可用，这里做兼容性导入。
try:
    import collections.abc as collections_abc
except ImportError:
    import collections as collections_abc


def is_str(x):
    """判断输入是否为字符串实例。"""
    return isinstance(x, six.string_types)


def iter_cast(inputs, dst_type, return_type=None):
    """将可迭代对象的元素转换为指定类型。

    Args:
        inputs (Iterable): 输入对象。
        dst_type (type): 目标类型。
        return_type (type, optional): 若指定，则输出会被转换为该类型，否则返回迭代器。

    Returns:
        iterator 或指定类型: 转换后的对象。
    """
    if not isinstance(inputs, collections_abc.Iterable):
        raise TypeError("inputs must be an iterable object")
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = six.moves.map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """将可迭代对象的元素转换成一个指定类型元素的列表。

    是 :func:`iter_cast` 的便捷版本。
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """将可迭代对象的元素转换成一个指定类型元素的元组。

    是 :func:`iter_cast` 的便捷版本。
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """判断是否为某类型的序列。

    Args:
        seq (Sequence): 待检查的序列。
        expected_type (type): 期望的元素类型。
        seq_type (type, optional): 期望的序列类型。

    Returns:
        bool: 序列是否合法。
    """
    if seq_type is None:
        exp_seq_type = collections_abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """判断是否为某类型的列表。

    是 :func:`is_seq_of` 的便捷版本。
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """判断是否为某类型的元组。

    是 :func:`is_seq_of` 的便捷版本。
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """按给定的长度列表把一个列表切成若干子列表。

    Args:
        in_list (list): 待切分的列表。
        lens(int or list): 每个输出子列表的期望长度。

    Returns:
        list: 若干子列表组成的列表。
    """
    if not isinstance(lens, list):
        raise TypeError('"indices" must be a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError(
            "sum of lens and list length does not match: {} != {}".format(
                sum(lens), len(in_list)
            )
        )
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx : idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """把嵌套的列表拼成一个扁平列表。

    Args:
        in_list (list): 待合并的列表的列表。

    Returns:
        list: 拼接后的扁平列表。
    """
    return list(itertools.chain(*in_list))


def check_prerequisites(
    prerequisites,
    checker,
    msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
    "found, please install them first.",
):
    """装饰器工厂：检查前置依赖是否满足。

    Args:
        prerequisites (str or list[str]): 需要检查的前置依赖。
        checker (callable): 检查函数，满足时返回 True，否则返回 False。
        msg_tmpl (str): 含两个占位符的消息模板。

    Returns:
        decorator: 具体的装饰器。
    """

    def wrap(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = (
                [prerequisites] if isinstance(prerequisites, str) else prerequisites
            )
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(", ".join(missing), func.__name__))
                raise RuntimeError("Prerequisites not meet.")
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    """通过尝试导入判断 Python 包是否已安装。"""
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    """通过 shell 的 which 命令判断可执行文件是否存在。"""
    if subprocess.call("which {}".format(cmd), shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """装饰器：检查某些 Python 包是否已安装。

    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """装饰器：检查某些可执行文件是否已安装。

    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)
