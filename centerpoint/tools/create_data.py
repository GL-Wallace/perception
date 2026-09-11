"""数据预处理入口脚本。

以命令行方式调度 Waymo / nuScenes 数据预处理流程：生成逐帧 info pickle 文件，
并对训练集构建 ground truth 数据库（gt database）。脚本通过 fire 将模块中的
函数名映射为子命令，例如：

    python tools/create_data.py nuscenes_data_prep <root_path> <version>
    python tools/create_data.py waymo_data_prep <root_path> <split>

主要函数：
    - nuscenes_data_prep: nuScenes 数据预处理（生成 info，训练集额外生成 gt database）。
    - waymo_data_prep: Waymo 数据预处理（生成 info，train 划分额外生成 gt database）。

依赖 det3d.datasets.nuscenes.nusc_common 与 det3d.datasets.waymo.waymo_common
提供的 info 生成函数，以及 det3d.datasets.utils.create_gt_database 生成 gt 数据库。
"""
import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False):
    """生成 nuScenes 数据集 info 文件，并为训练集构建 gt database。

    Args:
        root_path (str): nuScenes 数据集根目录。
        version (str): 数据版本，如 'v1.0-trainval'。
        nsweeps (int): 每帧聚合的 sweep 数量（含当前帧）。
        filter_zero (bool): 是否过滤掉速度/标注无效的样本。
        virtual (bool): 是否使用虚拟（复制）目标扩充 gt database。
    """
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    if version == 'v1.0-trainval':
        # 仅训练验证版本需要为训练集构建 gt database
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
            virtual=virtual
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    """生成 Waymo 数据集 info 文件，并为 train 划分构建 gt database。

    Args:
        root_path (str): Waymo 数据集根目录。
        split (str): 数据划分，如 'train'、'val'、'test'。
        nsweeps (int): 每帧聚合的 sweep 数量（含当前帧）。
    """
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        # 仅训练划分需要构建 gt database（限定使用三类主要目标）
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    # fire 将本模块中的函数暴露为命令行子命令
    fire.Fire()
