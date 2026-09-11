"""闭包 Hook。

允许在运行时把某个可调用对象动态绑定为 Hook 的指定钩子方法，
用于临时注入自定义逻辑而无需新写 Hook 子类。

主要类：
    - ClosureHook: 把外部函数注册为某个钩子方法。
"""

from .hook import Hook


class ClosureHook(Hook):
    """动态注册闭包函数的 Hook。

    Args:
        fn_name (str): 要绑定的钩子方法名（必须是 Hook 已有方法）。
        fn (callable): 要绑定的可调用对象。
    """

    def __init__(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        # 直接替换实例上的钩子方法，覆盖基类的空实现
        setattr(self, fn_name, fn)
