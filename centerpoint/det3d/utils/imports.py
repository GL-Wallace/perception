# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""按路径导入模块的工具（兼容 Python 2/3）。

依据 torch._six.PY3 判断 Python 版本，Python 3 用 importlib 实现，
否则回退到 imp.load_source。
"""

import torch

if torch._six.PY3:
    import importlib
    import importlib.util
    import sys

    # 参考 https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    def import_file(module_name, file_path, make_importable=False):
        """按文件路径导入模块。

        Args:
            module_name: 模块名。
            file_path: 文件路径。
            make_importable: 是否注册到 sys.modules。

        Returns:
            module: 导入的模块对象。
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if make_importable:
            sys.modules[module_name] = module
        return module


else:
    import imp
    # Python 2 分支：使用 imp.load_source 加载。
    def import_file(module_name, file_path, make_importable=None):
        module = imp.load_source(module_name, file_path)
        return module
