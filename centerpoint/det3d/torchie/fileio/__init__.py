"""文件 IO 子包。

对外暴露统一的 load/dump 接口、各类文件 handler 以及文本解析工具，供配置加载、
checkpoint 元数据等场景复用。

主要导出：
    - load / dump / register_handler: 统一读写与扩展注册入口。
    - BaseFileHandler 及各具体 handler: 各格式的实际读写实现。
    - list_from_file / dict_from_file: 简单文本解析工具。
"""

from .io import load, dump, register_handler
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .parse import list_from_file, dict_from_file

__all__ = [
    "load",
    "dump",
    "register_handler",
    "BaseFileHandler",
    "JsonHandler",
    "PickleHandler",
    "YamlHandler",
    "list_from_file",
    "dict_from_file",
]
