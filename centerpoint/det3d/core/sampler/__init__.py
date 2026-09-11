"""det3d.core.sampler GT 采样子包。

包含两个模块：
    - preprocess: 数据库预处理(难度/点数过滤)与 GT 框的增广操作(加噪、翻转、缩放、旋转)。
    - sample_ops: DataBaseSamplerV2，从 GT 数据库中采样物体并粘贴到当前场景点云。

sample_ops 依赖 preprocess 提供的 BatchSampler、碰撞检测与增广函数。
"""

from . import preprocess
from . import sample_ops
