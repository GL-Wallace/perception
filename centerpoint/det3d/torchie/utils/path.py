"""路径与文件系统工具。

提供文件判断、目录创建、符号链接与目录扫描等通用路径操作，供配置加载、
checkpoint 保存等训练基础设施复用。同时兼容 Python 2/3 的路径处理差异。

主要函数：
    - is_filepath / fopen: 判断路径并统一打开 str 与 Path 对象。
    - check_file_exist: 校验文件是否存在。
    - mkdir_or_exist: 幂等地创建目录。
    - symlink: 创建（可覆盖的）符号链接。
    - scandir: 扫描目录下的文件。
"""

import os
import os.path as osp
import sys
from pathlib import Path

import six

from .misc import is_str

# Python 2 没有内置 FileNotFoundError，统一别名以保证兼容。
if sys.version_info <= (3, 3):
    FileNotFoundError = IOError
else:
    FileNotFoundError = FileNotFoundError


def is_filepath(x):
    if is_str(x) or isinstance(x, Path):
        return True
    else:
        return False


def fopen(filepath, *args, **kwargs):
    """统一打开 str 或 pathlib.Path 形式的文件。"""
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """校验文件存在，不存在则抛出 FileNotFoundError。"""
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    """若目录不存在则创建；已存在则静默跳过。

    空字符串（表示不指定目录）直接返回。
    """
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


def symlink(src, dst, overwrite=True, **kwargs):
    """创建符号链接，生成前可按需删除已存在的目标。"""
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def _scandir_py35(dir_path, suffix=None):
    """基于 os.scandir 的目录扫描（Python 3.5+）。"""
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        filename = entry.name
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def _scandir_py(dir_path, suffix=None):
    """基于 os.listdir 的目录扫描（Python 2 兼容）。"""
    for filename in os.listdir(dir_path):
        if not osp.isfile(osp.join(dir_path, filename)):
            continue
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def scandir(dir_path, suffix=None):
    """扫描目录，返回其中（可选按后缀过滤的）文件名。

    Args:
        dir_path (str): 目录路径。
        suffix (str 或 tuple[str], optional): 文件名后缀过滤条件。

    Returns:
        generator: 匹配的文件名迭代器。
    """
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    if sys.version_info >= (3, 5):
        return _scandir_py35(dir_path, suffix)
    else:
        return _scandir_py(dir_path, suffix)
