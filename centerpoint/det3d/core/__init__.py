"""det3d.core 核心工具库。

聚合整个中心点检测框架所需的底层通用工具，按用途划分为四个子包：
    - utils: 特征散布(scatter)、圆形 NMS(circle_nms)、分布式梯度同步、通用杂项。
    - bbox: 旋转框的角点/几何/IoU/NMS 等张量与 NumPy 版本算子。
    - input: 体素生成器，负责点云到体素编号与坐标的转换。
    - sampler: GT 数据库采样与点云增广操作。

通过 `from .xxx import *` 将子包导出到本命名空间，供下游(det3d.datasets、
det3d.models 等)直接 import det3d.core 使用。
"""

from .utils import *
from .bbox import *
from .input import *
from .sampler import *
