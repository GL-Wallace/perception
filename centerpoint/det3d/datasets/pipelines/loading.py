"""点云数据加载 pipeline 阶段。

本模块是数据 pipeline 的入口阶段，负责从磁盘读取点云与标注：
    - LoadPointCloudFromFile: 读取当前帧及其历史 sweep 点云，多 sweep 时按
      transform_matrix 对齐并把 time_lag 拼到特征尾部。
    - LoadPointCloudAnnotations: 把 info 中的 GT box/名称/速度等组织到 res 中。

点云特征通过 read_file/read_single_waymo 等辅助函数读取（训练前已离线预处理），
支持 nuScenes 与 Waymo 两种数据集，以及 nuScenes 的 virtual(point painting) 模式。
"""
import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os 
from ..registry import PIPELINES

def _dict_select(dict_, inds):
    """按索引列表对 dict 中所有（含嵌套 dict 的）数组进行筛选。

    Args:
        dict_ (dict): 待筛选的字典。
        inds (np.ndarray): 保留的索引。
    """
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, virtual=False):
    """读取某个 lidar 点云文件。

    nuScenes 的点云以 float32 二进制存储，reshape 为 [N,5] 后取前 num_point_feature
    维。virtual=True 时额外加载 point painting 的虚拟点并拼接。

    Args:
        path (str): 点云文件路径（.bin）。
        tries (int): 重试次数（未使用）。
        num_point_feature (int): 返回的点特征维度。
        virtual (bool): 是否加载 virtual（point painting）点云。

    Returns:
        np.ndarray: [N, num_point_feature] 点云。
    """
    if virtual:
        # WARNING: hard coded for nuScenes 
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        tokens = path.split('/')
        seg_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        data_dict = np.load(seg_path, allow_pickle=True).item()

        # remove reflectance as other virtual points don't have this value  
        virtual_points1 = data_dict['real_points'][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] 
        virtual_points2 = data_dict['virtual_points']

        points = np.concatenate([points, np.ones([points.shape[0], 15-num_point_feature])], axis=1)
        virtual_points1 = np.concatenate([virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
        virtual_points2 = np.concatenate([virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    移除距原点（lidar 中心）过近的点，避免自车上的噪声点干扰。

    Args:
        points (np.ndarray): [C, N] 点云（前三行为 x/y/z）。
        radius (float): 距离阈值，x/y 绝对值均小于该值的点被剔除。

    Returns:
        np.ndarray: 过滤后的点云。
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    """读取并变换单个历史 sweep（nuScenes）。

    读取点云后，用 sweep 的 transform_matrix 把该帧点云对齐到参考帧坐标，并构造
    time_lag 向量（均为该帧与参考帧的时间差）。

    Args:
        sweep (dict): 含 lidar_path/transform_matrix/time_lag 的 sweep 信息。
        virtual (bool): 是否加载 virtual 点云。

    Returns:
        tuple: (points_sweep.T, curr_times.T)，分别为 [N,C] 点云与 [N,1] 时间差。
    """
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        # 齐次变换：对 xyz 左乘 [4,4] 变换矩阵，将历史帧对齐到参考帧坐标
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    """读取单帧 Waymo 点云（由 waymo_decoder 解码出的字典）。

    对 intensity 做 tanh 归一化，并把 xyz 与 intensity/elongation 特征拼接为 [N,5]。

    Args:
        obj (dict): 含 lidars.points_xyz 与 lidars.points_feature 的对象。

    Returns:
        np.ndarray: [N, 5] 点云。
    """
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    """读取并变换单个历史 Waymo sweep。

    Args:
        sweep (dict): 含 path/transform_matrix/time_lag 的 sweep 信息。

    Returns:
        tuple: (points_sweep.T, curr_times.T)。
    """
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        # 齐次变换对齐到参考帧坐标
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T


def get_obj(path):
    """从 pickle 文件读出单个对象（帧点云或标注）。

    Args:
        path (str): pickle 文件路径。

    Returns:
        object: 反序列化得到的字典对象。
    """
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    """从文件加载点云（含历史 sweep 聚合）。

    对 nuScenes 从 nsweeps-1 个历史 sweep 中随机选取若干帧；对 Waymo 顺序取前
    nsweeps-1 个历史帧。多 sweep 时把对齐后的点云与 time_lag 拼接成 combined。

    Args:
        dataset (str): 数据集类型，决定点云读取方式。
        random_select (bool): 是否随机选取历史 sweep（仅 nuScenes 使用）。
        npoints (int): 采样点数上限（当前未使用）。
    """
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):
        """加载点云并写入 res["lidar"]。

        Args:
            res (dict): 结果字典（含有 metadata 与 lidar 占位）。
            info (dict): 样本 info（含 lidar 路径与 sweeps）。

        Returns:
            tuple: (res, info)。
        """

        res["type"] = self.type

        if self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])
        
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1: 
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    """从 info 加载点云标注（GT box），写入 res["lidar"]["annotations"]。

    nuScenes 额外提供速度与 token；Waymo 仅提供 boxes 与 names。
    """
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):
        """加载标注。

        Args:
            res (dict): 结果字典。
            info (dict): 样本 info（含 gt_boxes/gt_names 等）。

        Returns:
            tuple: (res, info)。
        """

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
