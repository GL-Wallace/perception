"""json 文件读写 handler。

基于标准库 json 实现文本配置与结构化数据的序列化与反序列化，常用于读取
json 格式的配置或检测结果。

主要类：
    - JsonHandler: 实现 json 格式的 load/dump。
"""

import json

from .base import BaseFileHandler


class JsonHandler(BaseFileHandler):
    """json 格式的读写 handler。"""

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)
