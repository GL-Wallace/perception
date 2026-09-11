"""统一文件 IO 接口。

将 json/yaml/pickle 等序列化格式的读写封装为一对高层函数 load 与 dump，并借助
``file_handlers`` 注册表在运行时按文件扩展名分发到对应的 handler。config 加载
（utils/config.py 的 Config.fromfile）与 checkpoint 相关配置文件读取都依赖本模块。

主要函数：
    - load: 从文件或文件对象中反序列化数据。
    - dump: 将对象序列化到字符串或文件。
    - register_handler: 以装饰器方式注册自定义格式的 handler。
"""

from pathlib import Path

from ..utils import is_list_of, is_str
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler

# 扩展名到 handler 实例的注册表，用于 load/dump 时的分发。
file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
}


def load(file, file_format=None, **kwargs):
    """从 json/yaml/pickle 文件中读取数据。

    该方法为读取序列化文件提供了统一接口，其底层会根据文件格式分发到对应的 handler。

    Args:
        file (str 或 :obj:`Path` 或 file-like object): 文件名或文件对象。
        file_format (str, optional): 若未指定，则从文件扩展名推断格式；否则使用
            指定格式。当前支持的格式包括 "json"、"yaml/yml" 与 "pickle/pkl"。

    Returns:
        文件中的内容，类型取决于具体格式。
    """
    if isinstance(file, Path):
        file = str(file)
    # 未显式指定格式时，从路径末尾的扩展名推断。
    if file_format is None and is_str(file):
        file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError("Unsupported format: {}".format(file_format))

    # 按格式取出对应的 handler，并依据入参类型选择从路径还是文件对象读取。
    handler = file_handlers[file_format]
    if is_str(file):
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """将数据序列化为 json/yaml/pickle 字符串或文件。

    该方法为序列化数据提供了统一接口，可写入字符串或文件，并支持向各格式透传
    自定义参数。

    Args:
        obj (any): 待序列化的 Python 对象。
        file (str 或 :obj:`Path` 或 file-like object, optional): 若未指定，则将对象
            序列化为字符串；否则写入由文件名或文件对象指定的目标。
        file_format (str, optional): 同 :func:`load`。

    Returns:
        bool: 成功返回 True，否则返回 False。
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError("Unsupported format: {}".format(file_format))

    # 取出对应 handler；file 为 None 时转字符串，否则按路径或文件对象写入。
    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler, file_formats):
    """为某些扩展名注册一个 handler。

    Args:
        handler (:obj:`BaseFileHandler`): 待注册的 handler。
        file_formats (str 或 list[str]): 该 handler 负责处理的扩展名列表。
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            "handler must be a child of BaseFileHandler, not {}".format(type(handler))
        )
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError("file_formats must be a str or a list of str")
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):
    """类装饰器：将被装饰的 handler 类实例化并注册到指定扩展名。

    用法示例::

        @register_handler("txt")
        class TxtHandler(BaseFileHandler):
            ...
    """
    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
