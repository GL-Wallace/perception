"""Waymo 数据集的 PyTorch Dataset 封装。

WaymoDataset 将预处理得到的 infos pickle 组织为可迭代数据集：通过 pipeline
完成点云加载、多 sweep 聚合、体素化与标签分配，并为训练/评测提供统一接口。

每个样本由 get_sensor_data 构造 res 字典并送入 pipeline；评测时调用
waymo_common 将检测结果转换为 Waymo 官方评测格式。
"""
import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class WaymoDataset(PointCloudDataset):
    """Waymo 3D 点云数据集，继承自 PointCloudDataset。

    单 sweep 时每个点含 5 维特征 (x, y, z, intensity, elongation)；多 sweep 时
    额外追加一维 time_lag，因此特征维数为 6。
    """
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
        self,
        info_path,
        root_path,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        """初始化数据集。

        Args:
            info_path (str): 由 waymo_common 生成的 infos pickle 路径。
            root_path (str): 数据集根目录。
            cfg: 配置对象（本类中未直接使用）。
            pipeline (list): 数据预处理 pipeline 配置列表。
            class_names (list): 关心的类别名列表。
            test_mode (bool): 是否评测模式（决定样本 mode 字段）。
            sample (bool): 是否采样（保留参数，未使用）。
            nsweeps (int): 聚合的 sweep 数量。
            load_interval (int): 加载时下采样间隔（每 load_interval 帧取 1 帧）。
        """
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self._info_path = info_path
        self._class_names = class_names
        # 多 sweep 时追加 time_lag 维度
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1

    def reset(self):
        """重置数据集（Waymo 数据集未实现采样，直接断言失败）。"""
        assert False 

    def load_infos(self, info_path):
        """从 pickle 加载帧 info 列表，并按 load_interval 下采样。

        Args:
            info_path (str): infos pickle 路径。
        """

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        self._waymo_infos = _waymo_infos_all[::self.load_interval]

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):
        """返回数据集帧数（首次访问时惰性加载 infos）。"""

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        """构造第 idx 帧的 res 字典并跑完 pipeline，返回模型输入。

        Args:
            idx (int): 帧索引。

        Returns:
            dict: 经 pipeline 处理后的数据（含 voxels、targets 或 points 等）。
        """
        info = self._waymo_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False):
        """把检测结果写为 Waymo 官方评测格式（具体评测由 waymo devkit 完成）。

        Args:
            detections (dict): token -> 检测结果 的映射。
            output_dir (str): 结果输出目录，最终写入 detection_pred.bin。
            testset (bool): 是否测试集（本方法未使用该标记）。

        Returns:
            tuple: (None, None)，评测指标由外部 devkit 工具给出。
        """
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos 
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None 

