"""det3d.core.utils 工具子包。

汇总以下通用工具模块并统一导出：
    - dist_utils: 分布式梯度 allreduce 与优化器 hook。
    - misc: 张量转图像、多输入映射、反挂接等杂项函数。
    - center_utils: 中心点热图相关的高斯半径计算与绘制。
    - circle_nms_jit: 基于 numba 的圆形距离 NMS。

仅负责 import 导出，具体功能见各子模块。
"""

from .dist_utils import *
from .misc import *
from .center_utils import * 
from .circle_nms_jit import * 