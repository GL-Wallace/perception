"""点云数据集的抽象基类。

PointCloudDataset 定义所有点云数据集（Waymo/nuScenes 等）的公共接口：持有根目录、
pipeline（Compose 实例），并声明子类需实现的 __len__/__getitem__/get_sensor_data/
evaluation/ground_truth_annotations 等方法。
"""
import os.path as osp
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .registry import DATASETS
from .pipelines import Compose


@DATASETS.register_module
class PointCloudDataset(Dataset):
    """所有点云数据集应继承的抽象基类。

    子类需实现 ``__len__`` 与 ``__getitem__``（支持从 0 到 len-1 的整数索引）。
    """

    NumPointFeatures = -1
    CLASSES = None

    def __init__(
        self,
        root_path,
        info_path,
        pipeline=None,
        test_mode=False,
        class_names=None,
        **kwrags
    ):
        """初始化数据集。

        Args:
            root_path (str): 数据集根目录。
            info_path (str): infos pickle 路径。
            pipeline (list, optional): 数据预处理 pipeline 配置列表。
            test_mode (bool): 是否评测模式。
            class_names (list): 关心的类别名列表。
        """
        self._info_path = info_path
        self._root_path = Path(root_path)
        self._class_names = class_names

        self.test_mode = test_mode

        self._set_group_flag()

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)

    def __getitem__(self, index):
        """供预处理使用，需构造网络推理输入 dict（voxels/num_points/coordinates 等）。

        训练时需额外提供 labels/reg_targets；metadata 中可附带 image index 等信息。
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_sensor_data(self, query):
        """数据集需提供的统一取数接口。

        Args:
            query: int 或 dict。int 表示返回全部传感器数据；dict 表示按传感器查询。

        Returns:
            dict: 传感器数据（lidar 点云与标注等）与 metadata。
        """
        raise NotImplementedError

    def evaluation(self, dt_annos, output_dir):
        """数据集需提供的评测接口。"""
        raise NotImplementedError

    @property
    def ground_truth_annotations(self):
        """提供评测所需的 GT 标注（bbox/alpha/location/dimensions/rotation_y 等）。"""
        raise NotImplementedError

    def pre_pipeline(self, results):
        """（图像数据集预留）填充 results 的通用字段。"""
        results["img_prefix"] = self.img_prefix
        results["seg_prefix"] = self.seg_prefix
        results["proposal_file"] = self.proposal_file
        results["bbox_fields"] = []
        results["mask_fields"] = []

    def _filter_imgs(self, min_size=32):
        """过滤尺寸过小的图像（图像数据集预留）。"""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos):
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """设置 group 标记（当前实现为全 1，供分组采样器使用）。

        点云数据集不使用长宽比分组，因此所有样本标记为同一组。
        """
        self.flag = np.ones(len(self), dtype=np.uint8)
        # self.flag = np.zeros(len(self), dtype=np.uint8)
        # for i in range(len(self)):
        #     img_info = self.img_infos[i]
        #     if img_info['width'] / img_info['height'] > 1:
        #         self.flag[i] = 1

    def prepare_train_input(self, idx):
        """（预留）构造训练输入并跑 pipeline。"""
        raise NotImplementedError

        # img_info = self.img_infos[idx]
        # ann_info = self.get_ann_info(idx)
        # results = dict(img_info=img_info, ann_info=ann_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        # self.pre_pipeline(results)
        # return self.pipeline(results)

    def prepare_test_input(self, idx):
        """（预留）构造测试输入并跑 pipeline。"""
        raise NotImplementedError

        # img_info = self.img_infos[idx]
        # results = dict(img_info=img_info)
        # if self.proposals is not None:
        #     results['proposals'] = self.proposals[idx]
        # self.pre_pipeline(results)
        # return self.pipeline(results)
