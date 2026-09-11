"""det3d.models 包入口：统一导出全部网络组件与构建器。

按需导入 backbone（依赖 spconv）、bbox_heads、detectors、necks、readers、
second_stage、roi_heads 等子模块，并导出各构建接口与注册表。检测到 spconv
缺失时跳过稀疏卷积骨干并打印提示。
"""

import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None
if found:
    from .backbones import *  # noqa: F401,F403
else:
    print("No spconv, sparse convolution disabled!")
from .bbox_heads import *  # noqa: F401,F403
from .builder import (
    build_backbone,
    build_detector,
    build_head,
    build_loss,
    build_neck,
    build_roi_head
)
from .detectors import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .readers import *
from .registry import (
    BACKBONES,
    DETECTORS,
    HEADS,
    LOSSES,
    NECKS,
    READERS,
)
from .second_stage import * 
from .roi_heads import * 

__all__ = [
    "READERS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
    "build_detector",
]
