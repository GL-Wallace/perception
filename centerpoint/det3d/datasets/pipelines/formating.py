"""数据格式化 pipeline 阶段（Reformat）。

Reformat 是 pipeline 的最后一个阶段，把分散在 res 中的点云、voxels、targets 等
打包成模型可直接消费的 data_bundle 字典。评测且开启 double_flip 时，会额外为
y/x/xy 三种翻转版本各构造一个 data_bundle，返回列表供 TTA。
"""
from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    """简单的数据打包容器（当前实际使用原生 dict，此类保留作为类型标记）。"""

    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    """把 res 整理为模型输入格式。

    训练模式打包 voxels/coordinates/num_points/targets；评测模式打包原始点云，
    若开启 double_flip 则返回 [原始, yflip, xflip, double_flip] 四个 data_bundle。
    """

    def __init__(self, **kwargs):
        double_flip = kwargs.get('double_flip', False)
        self.double_flip = double_flip 

    def __call__(self, res, info):
        """执行格式化打包。

        Args:
            res (dict): pipeline 前置阶段累积的结果。
            info (dict): 样本 info。

        Returns:
            dict 或 list: 单个 data_bundle，或 double_flip 时的 data_bundle 列表。
        """
        meta = res["metadata"]
        points = res["lidar"]["points"]
        
        data_bundle = dict(
            metadata=meta
        )
        if points is not None:
            data_bundle.update(points=points)

        if 'voxels' in res["lidar"]:
            voxels = res["lidar"]["voxels"] 

            data_bundle.update(
                voxels=voxels["voxels"],
                shape=voxels["shape"],
                num_points=voxels["num_points"],
                num_voxels=voxels["num_voxels"],
                coordinates=voxels["coordinates"],
            )

        if res["mode"] == "train":
            data_bundle.update(res["lidar"]["targets"])
        elif res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

            if self.double_flip:
                # y axis 
                yflip_points = res["lidar"]["yflip_points"]
                yflip_voxels = res["lidar"]["yflip_voxels"] 
                yflip_data_bundle = dict(
                    metadata=meta,
                    points=yflip_points,
                    voxels=yflip_voxels["voxels"],
                    shape=yflip_voxels["shape"],
                    num_points=yflip_voxels["num_points"],
                    num_voxels=yflip_voxels["num_voxels"],
                    coordinates=yflip_voxels["coordinates"],
                )

                # x axis 
                xflip_points = res["lidar"]["xflip_points"]
                xflip_voxels = res["lidar"]["xflip_voxels"] 
                xflip_data_bundle = dict(
                    metadata=meta,
                    points=xflip_points,
                    voxels=xflip_voxels["voxels"],
                    shape=xflip_voxels["shape"],
                    num_points=xflip_voxels["num_points"],
                    num_voxels=xflip_voxels["num_voxels"],
                    coordinates=xflip_voxels["coordinates"],
                )
                # double axis flip 
                double_flip_points = res["lidar"]["double_flip_points"]
                double_flip_voxels = res["lidar"]["double_flip_voxels"] 
                double_flip_data_bundle = dict(
                    metadata=meta,
                    points=double_flip_points,
                    voxels=double_flip_voxels["voxels"],
                    shape=double_flip_voxels["shape"],
                    num_points=double_flip_voxels["num_points"],
                    num_voxels=double_flip_voxels["num_voxels"],
                    coordinates=double_flip_voxels["coordinates"],
                )

                return [data_bundle, yflip_data_bundle, xflip_data_bundle, double_flip_data_bundle], info


        return data_bundle, info



