"""det3d.core.bbox 旋转框几何子包。

包含三个模块：
    - geometry: 纯几何原语(凸多边形内点判断、线段相交、平面方程等)，主要用 numba 加速。
    - box_np_ops: NumPy 版旋转框/IoU/角点/点云增广等操作。
    - box_torch_ops: PyTorch 版旋转框张量算子与 PCDet 风格旋转框 NMS。

box_np_ops 依赖 geometry，box_torch_ops 相对独立，两者共同支撑数据预处理与后处理。
"""

from . import box_np_ops, box_torch_ops, geometry
