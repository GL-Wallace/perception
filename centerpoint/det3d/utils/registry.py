"""注册表（Registry）工具。

提供按名称注册类的 Registry 容器，以及通过配置字典构造对象的 build_from_cfg，
用于解耦配置与具体实现。
"""

import inspect

from det3d import torchie


class Registry(object):
    """类注册表：维护名称到类的映射。"""

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + "(name={}, items={})".format(
            self._name, list(self._module_dict.keys())
        )
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """按名称获取已注册的类，不存在返回 None。"""
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """注册一个类。

        Args:
            module_class (:obj:`nn.Module`): 待注册的类。
        """
        if not inspect.isclass(module_class):
            raise TypeError(
                "module must be a class, but got {}".format(type(module_class))
            )
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(
                "{} is already registered in {}".format(module_name, self.name)
            )
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        """装饰器形式的类注册，返回原类。"""
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """根据配置字典构造对象。

    Args:
        cfg (dict): 配置字典，至少包含 "type" 键。
        registry (:obj:`Registry`): 用于查找类型的注册表。
        default_args (dict, optional): 默认初始化参数。

    Returns:
        obj: 构造得到的对象。
    """
    assert isinstance(cfg, dict) and "type" in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop("type")
    if torchie.is_str(obj_type):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                "{} is not in the {} registry".format(obj_type, registry.name)
            )
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        # 默认参数只在未显式提供时生效。
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_cls(**args)
