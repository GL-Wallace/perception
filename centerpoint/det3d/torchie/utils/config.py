"""配置加载与访问工具。

提供训练配置的统一入口：支持从 python/yaml/json 文件加载配置，并以「字典 + 属性」
两种方式访问配置项，是训练主循环读取模型、数据、优化器等配置的核心基础设施。

主要类/函数：
    - ConfigDict: 支持属性访问的字典，访问缺失键时报错而非静默返回。
    - Config: 配置容器，负责文件的加载（.py 动态导入 / yaml/json 解析）、
        原始文本保留与属性/下标访问。
    - add_args: 将配置项自动注册为命令行参数。
"""

import os.path as osp
import sys
from argparse import ArgumentParser
from importlib import import_module

from addict import Dict

from .misc import collections_abc
from .path import check_file_exist


class ConfigDict(Dict):
    """支持属性访问的配置字典，访问缺失键时抛出 KeyError。"""

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )
        except Exception as e:
            ex = e
        else:
            return value
        # 将缺失键的 KeyError 转为 AttributeError，使属性访问语义更直观。
        raise ex


def add_args(parser, cfg, prefix=""):
    """把配置中的项递归注册为命令行参数。

    根据值的类型选择相应的 argparse 参数类型；嵌套 dict 会以点号拼接前缀递归展开，
    从而允许命令行覆盖配置项。
    """
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action="store_true")
        elif isinstance(v, dict):
            add_args(parser, v, k + ".")
        elif isinstance(v, collections_abc.Iterable):
            parser.add_argument("--" + prefix + k, type=type(v[0]), nargs="+")
        else:
            print("connot parse key {} of type {}".format(prefix + k, type(v)))
    return parser


class Config(object):
    """配置的容器与加载设施。

    支持将 python/json/yaml 作为配置文件。其使用方式与 dict 一致，同时允许以属性
    方式访问配置值。

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/torchie/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/torchie/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"

    """

    @staticmethod
    def fromfile(filename):
        """从配置文件加载并构造 Config。

        .py 文件通过动态导入执行后收集其中的全局变量；yml/yaml/json 文件则
        委托 fileio 的 load 解析。

        Args:
            filename (str): 配置文件路径。

        Returns:
            Config: 解析得到的配置对象。
        """
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        if filename.endswith(".py"):
            # 去掉 .py 后缀作为模块名，动态导入执行配置脚本。
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)
            # 临时把配置所在目录加入 sys.path，以支持配置内的相对导入。
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            # 收集模块中不以双下划线开头的全局变量作为配置项。
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }
        elif filename.endswith((".yml", ".yaml", ".json")):
            import torchie

            cfg_dict = torchie.load(filename)
        else:
            raise IOError("Only py/yml/yaml/json type are supported now!")
        return Config(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """自动根据配置文件生成 argparser（实验性）。"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        # 保留配置文件的原始文本，便于记录/展示所用配置。
        if filename:
            with open(filename, "r") as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __repr__(self):
        return "Config (path: {}): {}".format(self.filename, self._cfg_dict.__repr__())

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        # 写入的 dict 统一转成 ConfigDict，以保持属性访问能力。
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)
