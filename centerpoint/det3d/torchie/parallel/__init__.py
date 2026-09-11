"""并行训练基础设施入口。

聚合 DataContainer、collate 工具与 DataParallel / 分布式封装，对外提供
single-GPU/distributed 训练时数据整理与切分的统一接口。
"""

from .collate import collate, collate_kitti
from .data_container import DataContainer
from .data_parallel import MegDataParallel
from .distributed import MegDistributedDataParallel
from .scatter_gather import scatter, scatter_kwargs

__all__ = [
    "collate",
    "collate_kitti",
    "DataContainer",
    "MegDataParallel",
    "MegDistributedDataParallel",
    "scatter",
    "scatter_kwargs",
]
