"""文件 handler 的抽象基类。

定义统一读写接口的骨架：子类只需实现三个抽象方法
（load_from_fileobj / dump_to_fileobj / dump_to_str），即可接入 fileio 的
load/dump 分发。基于路径的两个方法已在此实现，会自动打开文件并委托给文件对象方法。

主要类：
    - BaseFileHandler: 所有格式 handler 的公共基类。
"""

from abc import ABCMeta, abstractmethod


class BaseFileHandler(object):
    """文件读写 handler 的抽象基类。

    子类必须实现从文件对象读取、写入文件对象以及序列化为字符串三个抽象方法；
    基于文件路径的读写直接打开文件后复用上述方法。
    """

    __metaclass__ = ABCMeta  # python 2 compatibility

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode="r", **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode="w", **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
