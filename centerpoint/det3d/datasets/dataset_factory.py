"""数据集类别名到数据集类的工厂映射。

提供简单的命名到实现类的查找，供 create_gt_database 等工具按字符串创建数据集。
"""
from .nuscenes import NuScenesDataset
from .waymo import WaymoDataset

# 数据集类别名 -> 数据集类
dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset
}


def get_dataset(dataset_name):
    """按类别名返回数据集类。

    Args:
        dataset_name (str): 数据集类别名（NUSC / WAYMO）。

    Returns:
        type: 对应的数据集类。
    """
    return dataset_factory[dataset_name]
