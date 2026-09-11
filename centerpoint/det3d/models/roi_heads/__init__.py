"""两阶段精炼 ROI 头子包（论文 Sec 3.4）。

导出 RoIHeadTemplate / RoIHead：基于第一阶段检测框在 BEV 特征上采样点特征，
经 MLP 预测 IoU 引导的置信度与 box 精炼。具体实现见 roi_head.py 与
roi_head_template.py。
"""
from .roi_head_template import RoIHeadTemplate
from .roi_head import RoIHead

__all__ = [
    'RoIHeadTemplate',
    'RoIHead'
]
