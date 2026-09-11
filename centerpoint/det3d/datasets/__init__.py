"""数据集子包入口。

对外暴露 NuScenesDataset/WaymoDataset、数据集包装器、采样器、DataLoader 构建函数
与 DATASETS 注册表，并统一提供 build_dataset。
"""
from .builder import build_dataset

# from .cityscapes import CityscapesDataset
from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset

# from .custom import CustomDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset

# from .extra_aug import ExtraAugmentation
from .loader import DistributedGroupSampler, GroupSampler, build_dataloader
from .registry import DATASETS

# from .voc import VOCDataset
# from .wider_face import WIDERFaceDataset
# from .xml_style import XMLDataset
#
__all__ = [
    "CustomDataset",
    "GroupSampler",
    "DistributedGroupSampler",
    "build_dataloader",
    "ConcatDataset",
    "RepeatDataset",
    "DATASETS",
    "build_dataset",
]
