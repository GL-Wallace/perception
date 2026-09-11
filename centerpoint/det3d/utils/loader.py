"""动态模块加载工具。

支持按文件路径导入 Python 模块：优先尝试通过 PYTHONPATH 进行常规 import，
失败则用 importlib 直接按路径加载，并可选注册到 sys.modules 以便反射查找。

主要函数：
    - _get_possible_module_path: 从 PYTHONPATH 中收集可作为模块的路径。
    - _get_regular_import_name: 计算路径对应的常规导入名。
    - import_file: 按路径导入文件。
    - import_name: 按名称导入模块的简单封装。
"""

import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("det3d.utils.loader")

CUSTOM_LOADED_MODULES = {}


def _get_possible_module_path(paths):
    """从给定路径列表收集可作为模块导入的路径（.py 或 .so 文件或目录）。"""
    ret = []
    for p in paths:
        p = Path(p)
        for path in p.glob("*"):
            if path.suffix in ["py", ".so"] or (path.is_dir()):
                if path.stem.isidentifier():
                    ret.append(path)
    return ret


def _get_regular_import_name(path, module_paths):
    """根据文件路径与可能的模块搜索路径推导常规导入名（点分模块名）。"""
    path = Path(path)
    for mp in module_paths:
        mp = Path(mp)
        if mp == path:
            return path.stem
        try:
            relative_path = path.relative_to(Path(mp))
            parts = list((relative_path.parent / relative_path.stem).parts)
            module_name = ".".join([mp.stem] + parts)
            return module_name
        except Exception:
            pass
    return None


def import_file(path, name: str = None, add_to_sys=True, disable_warning=False):
    """按文件路径导入模块。

    若能通过 PYTHONPATH 常规导入则直接导入；否则按路径加载。可选将模块注册到
    sys.modules，便于后续通过反射查找其中定义的对象。

    Args:
        path: 文件路径。
        name: 可选，指定模块名（默认取文件名 stem）。
        add_to_sys: 是否注册到 sys.modules。
        disable_warning: 是否关闭常规导入失败的告警。

    Returns:
        module: 导入得到的模块对象。
    """
    global CUSTOM_LOADED_MODULES
    path = Path(path)
    module_name = path.stem
    try:
        user_paths = os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        user_paths = []
    possible_paths = _get_possible_module_path(user_paths)
    model_import_name = _get_regular_import_name(path, possible_paths)
    if model_import_name is not None:
        return import_name(model_import_name)
    if name is not None:
        module_name = name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not disable_warning:
        logger.warning(
            (
                f"Failed to perform regular import for file {path}. "
                "this means this file isn't in any folder in PYTHONPATH "
                "or don't have __init__.py in that project. "
                "directly file import may fail and some reflecting features are "
                "disabled even if import succeed. please add your project to PYTHONPATH "
                "or add __init__.py to ensure this file can be regularly imported. "
            )
        )

    if add_to_sys:  # 注册到 sys.modules，便于反射查找文件中定义的对象。
        # 避免覆盖系统模块。
        if module_name in sys.modules and module_name not in CUSTOM_LOADED_MODULES:
            raise ValueError(f"{module_name} exists in system.")
        CUSTOM_LOADED_MODULES[module_name] = module
        sys.modules[module_name] = module
    return module


def import_name(name, package=None):
    """按名称导入模块的简单封装。"""
    module = importlib.import_module(name, package)
    return module
