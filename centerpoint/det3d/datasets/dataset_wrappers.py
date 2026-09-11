"""数据集包装器（ConcatDataset / RepeatDataset）。

对多个数据集做拼接或对单个数据集做重复，兼容 group flag 的拼接，减少小数据集在
epoch 间的数据加载开销。
"""
import numpy as np
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .registry import DATASETS


@DATASETS.register_module
class ConcatDataset(_ConcatDataset):
    """拼接多个数据集，并拼接各自的 group flag。

    Args:
        datasets (list): 数据集列表。
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], "flag"):
            flags = []
            for i in range(0, len(datasets)):
                flags.append(datasets[i].flag)
            self.flag = np.concatenate(flags)


@DATASETS.register_module
class RepeatDataset(object):
    """把数据集重复 times 次。

    适用于数据集较小但加载耗时的场景，跨 epoch 复用同一数据集，减少加载时间。

    Args:
        dataset: 被重复的数据集。
        times (int): 重复次数。
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        if hasattr(self.dataset, "flag"):
            self.flag = np.tile(self.dataset.flag, times)

        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
