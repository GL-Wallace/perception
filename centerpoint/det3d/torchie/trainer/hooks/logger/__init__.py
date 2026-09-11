"""日志 Hook 包入口。

向外暴露日志 Hook 基类及其三种实现：文本日志、TensorBoard 日志与 Pavi 日志，
统一供 Trainer.register_logger_hooks 按配置实例化。
"""

from .base import LoggerHook
from .pavi import PaviLoggerHook
from .tensorboard import TensorboardLoggerHook
from .text import TextLoggerHook

__all__ = ["LoggerHook", "TextLoggerHook", "PaviLoggerHook", "TensorboardLoggerHook"]
