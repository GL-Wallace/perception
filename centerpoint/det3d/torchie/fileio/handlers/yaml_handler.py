"""yaml 文件读写 handler。

基于 PyYAML 实现 yaml/yml 配置的读写，是训练配置解析的核心依赖。优先使用 C 扩展的
CLoader/CDumper 以提升性能，不可用时回退到纯 Python 实现。

主要类：
    - YamlHandler: 实现 yaml 格式的 load/dump。
"""

import yaml

# 优先使用 C 扩展加速 yaml 的解析与序列化；导入失败则回退到 Python 实现。
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):
    """yaml 格式的读写 handler。"""

    def load_from_fileobj(self, file, **kwargs):
        # 默认使用 C 版 Loader，除非调用方显式覆盖。
        kwargs.setdefault("Loader", Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault("Dumper", Dumper)
        return yaml.dump(obj, **kwargs)
