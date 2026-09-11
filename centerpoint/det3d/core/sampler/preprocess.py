"""GT 数据库预处理与点云/框增广操作。

提供两大部分能力：
    1. 数据库预处理：按难度、点数过滤 GT 数据库，以及按顺序编排多个预处理器；
    2. 增广操作：对 GT 框与点云做随机旋转、平移、缩放、翻转、按物体加噪等，
       并为 GT 采样提供碰撞检测(box_collision_test)与视锥裁剪相关辅助。

主要类/函数：
    - BatchSampler: 带洗牌与重置的采样器，按顺序从列表抽样。
    - DataBasePreprocessor / DBFilterByDifficulty / DBFilterByMinNumPoint:
        GT 数据库的过滤与预处理编排。
    - filter_gt_box_outside_range(_by_center) / filter_gt_low_points: 过滤框。
    - noise_per_object_v2_ / noise_per_object_v3_: 逐物体随机加噪(核心增广)。
    - global_scaling / global_rotation / random_flip(_both) 等全局增广。
    - box_collision_test: 旋转框两两碰撞检测(供 paste 采样避免重叠)。

依赖 det3d.core.bbox 的 box_np_ops 与 geometry 提供角点/内点/碰撞几何原语。
"""
import abc
import sys
import time
from collections import OrderedDict
from functools import reduce

import numba
import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import (
    is_line_segment_intersection_jit,
    points_in_convex_polygon_3d_jit,
    points_in_convex_polygon_jit,
)
import copy


class BatchSampler:
    """带洗牌与游标的列表采样器。

    维护一份索引数组，按游标顺序切片抽样；可选洗牌，抽到底部时自动重置
    (并重新洗牌)。

    Attributes:
        _sampled_list: 被采样的元素列表。
        _indices: 元素索引数组(洗牌后即抽样顺序)。
        _idx: 当前抽样游标。
        _shuffle: 是否在初始与重置时洗牌。
        _drop_reminder: 是否丢弃尾部不足一批的余数(当前未实际使用)。
    """

    def __init__(
        self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """按游标取 num 个索引；若越界则返回剩余部分并重置游标。"""
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """重置游标，(可选)重新洗牌进入下一轮。"""
        # if self._name is not None:
        #     print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """抽样 num 个实际元素。

        Args:
            num (int): 抽样数量。

        Returns:
            list: 抽样得到的元素列表。
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]
        # return np.random.choice(self._sampled_list, num)


class DataBasePreprocessing:
    """GT 数据库预处理器的抽象基类。

    子类实现 _preprocess 完成对 db_infos 的处理；通过 __call__ 统一被调用。
    """

    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass


class DBFilterByDifficulty(DataBasePreprocessing):
    """按难度过滤 GT 数据库中的样本。

    移除 difficulty 出现在 removed_difficulties 中的样本。
    """

    def __init__(self, removed_difficulties, logger=None):
        self._removed_difficulties = removed_difficulties
        logger.info(f"{removed_difficulties}")

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info
                for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


class DBFilterByMinNumPoint(DataBasePreprocessing):
    """按框内点数下限过滤 GT 数据库样本。

    min_gt_point_dict 给出各类别最少点数，低于下限的样本被剔除。
    """

    def __init__(self, min_gt_point_dict, logger=None):
        self._min_gt_point_dict = min_gt_point_dict
        logger.info(f"{min_gt_point_dict}")

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


class DataBasePreprocessor:
    """按顺序编排多个预处理器并依次作用到 db_infos。"""

    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos


def filter_gt_box_outside_range(gt_boxes, limit_range):
    """过滤训练范围之外的 GT 框(按框整体是否超出 range)。

    此函数应在其它预处理函数之后调用。若框的任意角点落在 range 内则保留。

    Args:
        gt_boxes (np.ndarray): [N, 7] GT 框。
        limit_range (list): [xmin, ymin, zmin, xmax, ymax, zmax] 范围。

    Returns:
        np.ndarray: [N] bool 掩码，True 表示保留。
    """
    gt_boxes_bv = box_np_ops.center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, -1]
    )
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_boxes_bv.reshape(-1, 2), bounding_box)
    return np.any(ret.reshape(-1, 4), axis=1)


def filter_gt_box_outside_range_by_center(gt_boxes, limit_range):
    """过滤中心在训练范围之外的 GT 框。

    此函数应在其它预处理函数之后调用。仅保留中心落在 range 内的框。

    Args:
        gt_boxes (np.ndarray): [N, 7] GT 框。
        limit_range (list): [xmin, ymin, zmin, xmax, ymax, zmax] 范围。

    Returns:
        np.ndarray: [N] bool 掩码。
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = box_np_ops.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    ret = points_in_convex_polygon_jit(gt_box_centers, bounding_box)
    return ret.reshape(-1)


def filter_gt_low_points(gt_boxes, points, num_gt_points, point_num_threshold=2):
    """过滤含点过少的 GT 框，并同时移除这些框内的点。

    Args:
        gt_boxes (np.ndarray): [N, 7] GT 框。
        points (np.ndarray): [M, ...] 点云。
        num_gt_points (np.ndarray): [N] 每个框包含的点数。
        point_num_threshold (int): 保留框所需的最少点数阈值。

    Returns:
        tuple: (过滤后的 gt_boxes, 过滤后的 points)。
    """
    points_mask = np.ones([points.shape[0]], np.bool)
    gt_boxes_mask = np.ones([gt_boxes.shape[0]], np.bool)
    for i, num in enumerate(num_gt_points):
        if num <= point_num_threshold:
            masks = box_np_ops.points_in_rbbox(points, gt_boxes[i : i + 1])
            masks = masks.reshape([-1])
            # 从点云中去掉该框内的点，并剔除该框
            points_mask &= np.logical_not(masks)
            gt_boxes_mask[i] = False
    return gt_boxes[gt_boxes_mask], points[points_mask]


def mask_points_in_corners(points, box_corners):
    """返回位于给定 3D 框角点区域内的点掩码。

    Args:
        points (np.ndarray): [M, >=3] 点云。
        box_corners (np.ndarray): [N, 8, 3] 框角点。

    Returns:
        np.ndarray: [M, N] bool 掩码。
    """
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return mask


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    """就地构造绕指定轴的 3x3 旋转矩阵(写入 rot_mat_T)。

    Args:
        rot_mat_T (np.ndarray): [3, 3] 输出矩阵。
        angle (float): 旋转角。
        axis (int): 旋转轴。
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    """就地绕原点旋转 2D 角点(旋转矩阵写到 rot_mat_T 供复用)。

    Args:
        corners (np.ndarray): [N, 2] 输出角点。
        angle (float): 旋转角。
        rot_mat_T (np.ndarray): [2, 2] 复用的旋转矩阵缓冲。
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def _box_single_to_corner_jit(boxes):
    """把 [x, y, dx, dy, yaw] 框转为 4 个 2D 角点(jit 版)。

    Args:
        boxes (np.ndarray): [N, 5] 框。

    Returns:
        np.ndarray: [N, 4, 2] 角点。
    """
    num_box = boxes.shape[0]
    # 构造相对中心的单位框角点(归一化坐标)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    """逐个框尝试加噪声，选出与其它框无碰撞的首个可行噪声。

    Args:
        boxes (np.ndarray): [N, 5] 框。
        valid_mask (np.ndarray): [N] 是否参与加噪。
        loc_noises (np.ndarray): [N, M, 3] 每个框 M 组位置噪声(仅前 2 维用于 BEV)。
        rot_noises (np.ndarray): [N, M] 每个框 M 组旋转噪声。

    Returns:
        np.ndarray: [N] 每框被选中的噪声尝试编号，-1 表示未选中。
    """
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                # 先把框平移回原点，旋转，再平移到加噪后的位置
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    """按组给框加噪声(整组共享同一个噪声尝试，并忽略组内自碰撞)。

    注意:
        此函数要求 boxes 已按 group id 排序。

    Args:
        boxes (np.ndarray): [N, 5] 框。
        valid_mask (np.ndarray): [N]。
        loc_noises (np.ndarray): [N, M, 3]。
        rot_noises (np.ndarray): [N, M]。
        group_nums (np.ndarray): 每组的框数量。

    Returns:
        np.ndarray: [N] 选中噪声编号。
    """
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_groups = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(
                        current_corners[i], rot_noises[idx + i, j], rot_mat_T
                    )
                    current_corners[i] += (
                        boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
                    )
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners
                )
                for i in range(num):  # remove self-coll
                    # 忽略本组内部的自碰撞(同组框允许重叠)
                    coll_mat[i, idx : idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_group_v2_(
    boxes, valid_mask, loc_noises, rot_noises, group_nums, global_rot_noises
):
    """按组加噪声(带全局旋转)，把噪声作用的结果写回 loc/rot_noises。

    注意:
        此函数要求 boxes 已按 group id 排序。与 noise_per_box_group 不同，
        本版本额外引入绕原点的全局旋转(global_rot_noises)，并把最终等效的
        位置与角度变化回写到 loc_noises/rot_noises 供后续应用。

    Args:
        boxes (np.ndarray): [N, 5] 框。
        valid_mask (np.ndarray): [N]。
        loc_noises (np.ndarray): [N, M, 3]。
        rot_noises (np.ndarray): [N, M]。
        group_nums (np.ndarray): 每组框数。
        global_rot_noises (np.ndarray): [N, M] 全局旋转噪声。

    Returns:
        np.ndarray: [N] 选中噪声编号。
    """
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)

    current_grot = np.zeros((max_group_num,), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num,), dtype=boxes.dtype)

    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)

    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    # 由极坐标计算全局旋转后的目标位置
                    current_radius = np.sqrt(
                        current_box[0, 0] ** 2 + current_box[0, 1] ** 2
                    )
                    current_grot[i] = np.arctan2(current_box[0, 0], current_box[0, 1])
                    dst_grot[i] = current_grot[i] + global_rot_noises[idx + i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    # 全局旋转同时改变框的朝向角
                    current_box[0, -1] += dst_grot[i] - current_grot[i]

                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_T[0, 0] = rot_cos
                    rot_mat_T[0, 1] = -rot_sin
                    rot_mat_T[1, 0] = rot_sin
                    rot_mat_T[1, 1] = rot_cos
                    current_corners[i] = (
                        current_box[0, 2:4] * corners_norm @ rot_mat_T
                        + current_box[0, :2]
                    )
                    current_corners[i] -= current_box[0, :2]

                    _rotation_box2d_jit_(
                        current_corners[i], rot_noises[idx + i, j], rot_mat_T
                    )
                    current_corners[i] += (
                        current_box[0, :2] + loc_noises[i + idx, j, :2]
                    )
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners
                )
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx : idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        # 把全局旋转引起的变化并入噪声，供后续统一应用
                        loc_noises[i + idx, j, :2] += dst_pos[i] - boxes[i + idx, :2]
                        rot_noises[i + idx, j] += dst_grot[i] - current_grot[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises, global_rot_noises):
    """逐框加噪声(带全局旋转)，并把等效变化回写到噪声数组。

    Args:
        boxes (np.ndarray): [N, 5] 框。
        valid_mask (np.ndarray): [N]。
        loc_noises (np.ndarray): [N, M, 3]。
        rot_noises (np.ndarray): [N, M]。
        global_rot_noises (np.ndarray): [N, M] 全局旋转噪声。

    Returns:
        np.ndarray: [N] 选中噪声编号。
    """
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                # 由极坐标计算全局旋转后的目标位置
                current_radius = np.sqrt(boxes[i, 0] ** 2 + boxes[i, 1] ** 2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += dst_grot - current_grot

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                )
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners
                )
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += dst_pos - boxes[i, :2]
                    rot_noises[i, j] += dst_grot - current_grot
                    break
    return success_mask


@numba.njit
def points_transform_(
    points, centers, point_masks, loc_transform, rot_transform, valid_mask
):
    """按框的刚体变换就地变换框内的点(每个点只应用第一个所属框的变换)。

    Args:
        points (np.ndarray): [M, >=3] 点云。
        centers (np.ndarray): [N, 3] 框中心。
        point_masks (np.ndarray): [M, N] 点是否落在每个框内。
        loc_transform (np.ndarray): [N, 3] 位置平移量。
        rot_transform (np.ndarray): [N] 旋转角(z 轴)。
        valid_mask (np.ndarray): [N] 有效框掩码。
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    # 平移到原点 -> 旋转 -> 平移回中心 -> 加位置平移
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = points[i : i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    """就地给有效框加位置平移与旋转角增量。

    Args:
        boxes (np.ndarray): [N, 7] 框。
        loc_transform (np.ndarray): [N, 3] 平移量。
        rot_transform (np.ndarray): [N] 旋转角增量。
        valid_mask (np.ndarray): [N] 有效框掩码。
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    """按选中的噪声编号从噪声候选中取出对应项。

    Args:
        transform (np.ndarray): [N, M, ...] 噪声候选。
        indices (np.ndarray): [N] 每行选中的编号，-1 表示未选中(取 0)。

    Returns:
        np.ndarray: [N, *transform.shape[2:]] 选出的噪声。
    """
    result = np.zeros((transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def group_transform_(loc_noise, rot_noise, locs, rots, group_center, valid_mask):
    """把每框的独立旋转噪声折算为绕组中心的位移噪声。

    当一组框绕其组中心旋转时，框中心位置会绕组中心发生弧线位移，本函数把
    这种位移折算到 loc_noise 中。

    Args:
        loc_noise (np.ndarray): [N, M, 3] 位置噪声(就地更新前 2 维)。
        rot_noise (np.ndarray): [N, M] 旋转噪声。
        locs (np.ndarray): [N, 3] 框中心。
        rots (np.ndarray): [N] 框朝向(未使用)。
        group_center (np.ndarray): [N, 3] 每框所属组的中心。
        valid_mask (np.ndarray): [N]。
    """
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            # calculate rots related to group center
            # 框相对组中心的方位角
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center)
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center)
                )


@numba.njit
def group_transform_v2_(
    loc_noise, rot_noise, locs, rots, group_center, grot_noise, valid_mask
):
    """group_transform_ 的带全局旋转版本。

    在绕组中心旋转的同时叠加全局旋转增量 grot_noise，把二者共同引起的
    位移折算到 loc_noise 中。
    """
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + grot_noise[i, j])
                    - np.sin(rot_center + grot_noise[i, j])
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + grot_noise[i, j])
                    - np.cos(rot_center + grot_noise[i, j])
                )


def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    """让同组框共享同一组位置/旋转噪声(取组内首框的噪声)。

    Args:
        loc_noise (np.ndarray): [N, M, 3] 位置噪声。
        rot_noise (np.ndarray): [N, M] 旋转噪声。
        group_ids (np.ndarray): [N] 每框的组 id。
    """
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    """让同组框共享位置/旋转/全局旋转噪声(取组内首框的噪声)。"""
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]
        grot_noise[i] = grot_noise[gid_to_index_dict[group_ids[i]]]


def get_group_center(locs, group_ids):
    """计算每个组(按 gid>=0)的中心坐标。

    Args:
        locs (np.ndarray): [N, 3] 框中心。
        group_ids (np.ndarray): [N] 每框的组 id，负数表示不属于任何组。

    Returns:
        tuple: (group_centers_ret, group_id_num_dict)，前者 [N, 3] 每框所属
            组的中心，后者记录每组框数量。
    """
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[i] = (
            group_centers[group_id_dict[gid]] / group_id_num_dict[gid]
        )
    return group_centers_ret, group_id_num_dict


def noise_per_object_v3_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=5,
    group_ids=None,
):
    """对每个 GT 框独立地随机旋转/平移(支持分组与全局旋转)。

    为每个框生成多组位置/旋转噪声候选，逐一尝试并做碰撞检测，选出可行的
    一组后应用到框及其内部点云。若传入 group_ids 则同组框共享噪声并绕组
    中心整体变换。

    Args:
        gt_boxes (np.ndarray): [N, 7] 雷达坐标系 GT 框。
        points (np.ndarray): [M, 4] 雷达点云(可为 None)。
        valid_mask (np.ndarray): [N] 有效框掩码。
        rotation_perturb (float 或 list): 旋转扰动范围。
        center_noise_std (float 或 list): 中心位置噪声标准差(xyz 三轴)。
        global_random_rot_range (float 或 list): 全局随机旋转范围。
        num_try (int): 每个框尝试的噪声组数。
        group_ids (np.ndarray): [N] 组 id，可为 None 表示逐框独立。
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    # 全局旋转范围非零时才启用全局旋转
    enable_grot = (
        np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    )
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    # 采样中心位置噪声(高斯)与旋转噪声(均匀)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    # 以框中心方位角为基准，把全局旋转范围换算成增量区间后采样
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try],
    )
    if group_ids is not None:
        if enable_grot:
            set_group_noise_same_v2_(
                loc_noises, rot_noises, global_rot_noises, group_ids
            )
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                group_centers,
                global_rot_noises,
                valid_mask,
            )
        else:
            group_transform_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                group_centers,
                valid_mask,
            )
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0.5]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )
    if group_ids is not None:
        if not enable_grot:
            selected_noise = noise_per_box_group(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                group_nums,
            )
        else:
            selected_noise = noise_per_box_group_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                group_nums,
                global_rot_noises,
            )
    else:
        if not enable_grot:
            selected_noise = noise_per_box(
                gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
            )
        else:
            selected_noise = noise_per_box_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]],
                valid_mask,
                loc_noises,
                rot_noises,
                global_rot_noises,
            )
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def noise_per_object_v2_(
    gt_boxes,
    points=None,
    valid_mask=None,
    rotation_perturb=np.pi / 4,
    center_noise_std=1.0,
    global_random_rot_range=np.pi / 4,
    num_try=100,
):
    """对每个 GT 框独立随机旋转/平移(支持全局旋转，不带分组)。

    与 noise_per_object_v3_ 逻辑相似，但不支持 group_ids；通过绕原点的全局
    旋转把框随机放置在圆周上的位置。

    Args:
        gt_boxes (np.ndarray): [N, 7] 雷达框。
        points (np.ndarray): [M, 4] 点云(可为 None)。
        valid_mask (np.ndarray): [N]。
        rotation_perturb (float 或 list): 旋转扰动范围。
        center_noise_std (float 或 list): 中心噪声标准差。
        global_random_rot_range (float 或 list): 全局旋转范围。
        num_try (int): 每框尝试次数。
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [-global_random_rot_range, global_random_rot_range]

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes,), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try]
    )
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try],
    )

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=origin, axis=2
    )
    if np.abs(global_random_rot_range[0] - global_random_rot_range[1]) < 1e-3:
        selected_noise = noise_per_box(
            gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises, rot_noises
        )
    else:
        selected_noise = noise_per_box_v2_(
            gt_boxes[:, [0, 1, 3, 4, 6]],
            valid_mask,
            loc_noises,
            rot_noises,
            global_rot_noises,
        )
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    if points is not None:
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def global_scaling(gt_boxes, points, scale=0.05):
    """对点云与 GT 框做随机整体缩放。

    比例在 [1-scale, 1+scale] 均匀采样。只缩放点的 xyz 与框的尺寸。

    Args:
        gt_boxes (np.ndarray): [N, >=6] 框。
        points (np.ndarray): [M, >=3] 点云。
        scale (float 或 list): 缩放扰动幅度。

    Returns:
        tuple: (gt_boxes, points)。
    """
    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    """对点云与 GT 框做随机整体绕 z 轴旋转。

    Args:
        gt_boxes (np.ndarray): [N, >=7] 框。
        points (np.ndarray): [M, >=3] 点云。
        rotation (float 或 list): 旋转范围。

    Returns:
        tuple: (gt_boxes, points)。
    """
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    if gt_boxes.shape[1] > 7:
        # 若框带速度(vx, vy)，同样绕 z 轴旋转速度方向
        gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
            np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
            noise_rotation,
            axis=2,
        )[:, :2]
    gt_boxes[:, -1] += noise_rotation
    return gt_boxes, points


def random_flip(gt_boxes, points, probability=0.5):
    """按概率对点云与 GT 框做 y 轴随机翻转。

    Args:
        gt_boxes (np.ndarray): [N, >=7] 框。
        points (np.ndarray): [M, >=3] 点云。
        probability (float): 翻转概率。

    Returns:
        tuple: (gt_boxes, points)。
    """
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        # y 翻转让 yaw 关于 pi/2 对称
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    return gt_boxes, points

def random_flip_both(gt_boxes, points, probability=0.5, flip_coor=None):
    """对点云与 GT 框做 x、y 两个方向独立的随机翻转。

    Args:
        gt_boxes (np.ndarray): [N, >=7] 框。
        points (np.ndarray): [M, >=3] 点云。
        probability (float): 每个方向的翻转概率。
        flip_coor (float): x 翻转的对称轴坐标，None 表示关于原点翻转。

    Returns:
        tuple: (gt_boxes, points)。
    """
    # x flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, -1] = -gt_boxes[:, -1] + np.pi
        points[:, 1] = -points[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    
    # y flip 
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        if flip_coor is None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            points[:, 0] = -points[:, 0]
        else:
            # 关于直线 x = flip_coor 对称
            gt_boxes[:, 0] = flip_coor * 2 - gt_boxes[:, 0]
            points[:, 0] = flip_coor * 2 - points[:, 0]

        gt_boxes[:, -1] = -gt_boxes[:, -1] + 2*np.pi  # TODO: CHECK THIS 
        
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, vx, vy, r
            gt_boxes[:, 6] = -gt_boxes[:, 6]
    
    return gt_boxes, points


def global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    """整体缩放(显式给定缩放上下界)。只缩放点 xyz 与框除 yaw 外的字段。"""
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :-1] *= noise_scale
    return gt_boxes, points


def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4, max_rad=np.pi / 4):
    """整体绕 z 轴旋转(显式给定旋转角上下界)。"""
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, -1] += noise_rotation
    return gt_boxes, points


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """判断两组 2D 旋转框两两是否碰撞(相交或包含)。

    Args:
        boxes (np.ndarray): [N, 4, 2] 第一组框角点。
        qboxes (np.ndarray): [K, 4, 2] 第二组框角点。
        clockwise (bool): 角点是否为顺时针顺序。

    Returns:
        np.ndarray: [N, K] bool 数组，True 表示碰撞。

    说明:
        先用轴对齐外接框粗筛，再做边-边相交判断；不相交时进一步判断一个
        框是否被另一个完整包含。
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    # slices 用于构造每条边(相邻两个角点)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            # 先做外接框粗筛
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    # 边-边两两相交判断(跨立实验)
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (
                                D[0] - A[0]
                            )
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (
                                D[0] - B[0]
                            )
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    C[0] - A[0]
                                )
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (
                                    D[0] - A[0]
                                )
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        # 检查框 box 是否包含 qbox 的顶点
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            # 再检查 qbox 是否包含 box 的顶点
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


def global_translate_(gt_boxes, points, noise_translate_std):
    """对点云与 GT 框做随机整体平移。

    Args:
        gt_boxes (np.ndarray): [N, >=3] 框。
        points (np.ndarray): [M, >=3] 点云。
        noise_translate_std (float 或 list): 三轴平移噪声标准差。

    Returns:
        tuple: (gt_boxes, points)。
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [noise_translate_std, noise_translate_std, noise_translate_std]
        )
    if all([e == 0 for e in noise_translate_std]):
        return gt_boxes, points
    noise_translate = np.array(
        [
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[0], 1),
        ]
    ).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points


if __name__ == "__main__":
    bboxes = np.array(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.2, 0.2, 0.6, 0.6],
            [0.7, 0.7, 0.9, 0.9],
            [0.55, 0.55, 0.8, 0.8],
        ]
    )
    bbox_corners = box_np_ops.minmax_to_corner_2d(bboxes)
    print(bbox_corners.shape)
    print(box_collision_test(bbox_corners, bbox_corners))
