"""检测头子包。

导出 CenterHead：CenterPoint 的核心检测头，将 BEV 特征转为中心热图(hm)
与属性回归(reg/height/dim/rot/vel)，训练时计算损失、推理时解码并后处理。
具体实现见 .center_head。
"""
from .center_head import CenterHead

__all__ = ["CenterHead"]
