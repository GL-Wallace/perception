"""训练引擎包入口。

聚合 Trainer、Hook 体系、checkpoint 读写、日志缓冲与优先级等基础设施，
以统一的命名空间对外暴露，供训练脚本（如 det3d.torchie.apis.train）导入使用。
"""

from .checkpoint import (
    load_checkpoint,
    load_state_dict,
    save_checkpoint,
    weights_to_cpu,
)
from .hooks import (
    CheckpointHook,
    ClosureHook,
    DistSamplerSeedHook,
    Hook,
    IterTimerHook,
    LoggerHook,
    LrUpdaterHook,
    OptimizerHook,
    PaviLoggerHook,
    TensorboardLoggerHook,
    TextLoggerHook,
)
from .log_buffer import LogBuffer
from .parallel_test import parallel_test
from .priority import Priority, get_priority
from .trainer import Trainer
from .utils import (
    get_dist_info,
    get_host_info,
    get_time_str,
    master_only,
    obj_from_dict,
)

__all__ = [
    "Trainer",
    "LogBuffer",
    "Hook",
    "CheckpointHook",
    "ClosureHook",
    "LrUpdaterHook",
    "OptimizerHook",
    "IterTimerHook",
    "DistSamplerSeedHook",
    "LoggerHook",
    "TextLoggerHook",
    "PaviLoggerHook",
    "TensorboardLoggerHook",
    "load_state_dict",
    "load_checkpoint",
    "weights_to_cpu",
    "save_checkpoint",
    "parallel_test",
    "Priority",
    "get_priority",
    "get_host_info",
    "get_dist_info",
    "master_only",
    "get_time_str",
    "obj_from_dict",
]
