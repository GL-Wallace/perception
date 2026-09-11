"""测试阶段数据增强（double flip，双重翻转）pipeline 阶段。

DoubleFlip 在评测时对输入点云做 y 翻转、x 翻转、x+y 翻转，生成三份副本，与原始
点云一起送入网络，用于测试时增广（TTA）。
"""
from det3d import torchie

from ..registry import PIPELINES
from .compose import Compose


@PIPELINES.register_module
class DoubleFlip(object):
    """对点云生成 y/x/xy 三种翻转副本，供 TTA 使用。"""

    def __init__(self):
        pass

    def __call__(self, res, info):
        """生成三种翻转点云并写入 res["lidar"]。

        Args:
            res (dict): 结果字典，含 lidar.points。
            info (dict): 样本 info。

        Returns:
            tuple: (res, info)。
        """
        # y flip
        points = res["lidar"]["points"].copy()
        points[:, 1] = -points[:, 1]

        res["lidar"]['yflip_points'] = points

        # x flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]

        res["lidar"]['xflip_points'] = points

        # x y flip
        points = res["lidar"]["points"].copy()
        points[:, 0] = -points[:, 0]
        points[:, 1] = -points[:, 1]

        res["lidar"]["double_flip_points"] = points  

        return res, info 



