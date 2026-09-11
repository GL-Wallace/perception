"""pickle 文件读写 handler。

基于标准库 pickle 实现二进制对象的序列化与反序列化，用于保存/恢复训练中间状态、
缓存点云/标注等二进制数据。

主要类：
    - PickleHandler: 实现 pickle 格式的 load/dump。
"""

from six.moves import cPickle as pickle

from .base import BaseFileHandler


class PickleHandler(BaseFileHandler):
    """pickle 格式的读写 handler。

    注意：
        pickle 涉及二进制协议，路径读写使用二进制模式；默认使用 protocol 2 以
        提高兼容性。
    """

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(filepath, mode="rb", **kwargs)

    def dump_to_str(self, obj, **kwargs):
        # 默认固定 protocol=2，保证跨 Python 版本的兼容性。
        kwargs.setdefault("protocol", 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault("protocol", 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(obj, filepath, mode="wb", **kwargs)
