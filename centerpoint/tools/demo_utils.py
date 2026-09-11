"""BEV 可视化工具（源自 nuScenes devkit）。

提供 3D 框数据结构 Box 与投影、渲染辅助函数，用于将点云、真值框与检测结果
绘制成鸟瞰视角（BEV）图片，供 demo.py 等脚本调用。

主要内容：
    - view_points: 将 3D 点投影到 2D 平面。
    - _second_det_to_nusc_box: 将 centerpoint 检测结果转换为 Box 列表。
    - Box: 3D 框数据类，支持旋转、渲染为 matplotlib 或 OpenCV 图。
    - visual: 绘制单帧 BEV 图（点云 + 真值框 + 预测框）并保存为 png。
    - remove_close: 过滤掉离原点过近的点。

注意：
    本文件大部分代码拷贝自 nuScenes devkit，坐标约定以 nuScenes 为准
    （x 向前、y 向左、z 向上）。
"""

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from matplotlib import pyplot as plt 


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """将 3D 点映射到 2D 平面。

    先计算点与 view 矩阵的点积，约定投影结果落到前两个坐标轴上；随后可选地
    沿第三维做归一化。可用于透视投影与正交投影。

    Args:
        points (np.ndarray): 形状 (3, n) 的点矩阵，每个点 (x, y, z) 沿列排列。
        view (np.ndarray): 定义投影矩阵（n <= 4），投影后角点落到前两轴。
            - 透视投影：view 为 3x3 相机矩阵，normalize=True；
            - 带平移的正交投影：view 为 3x4 矩阵，normalize=False；
            - 无平移的正交投影：view 为 3x3（或末列全 0 的 3x4），normalize=False。
        normalize (bool): 是否对剩余坐标（第三维）做归一化。

    Returns:
        np.ndarray: 形状 (3, n) 的映射后点；normalize=False 时第三维为高度。
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # 在齐次坐标下做投影运算
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def _second_det_to_nusc_box(detection):
    """将 centerpoint 检测结果转换为 nuScenes Box 列表。

    Args:
        detection (dict): 包含 box3d_lidar、scores、label_preds 的检测结果。

    Returns:
        List[Box]: 转换后的 Box 列表，包含位置、尺寸、朝向、类别、分数与速度。

    注意:
        这里将 lidar 坐标下的 yaw 统一旋转 -pi/2，对齐 nuScenes 的朝向约定。
    """
    box3d = detection["box3d_lidar"]
    scores = detection["scores"]
    labels = detection["label_preds"]
    box3d[:, -1] = -box3d[:, -1] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=box3d[i, -1])
        velocity = (*box3d[i, 6:8], 0.0)
        box = Box(
            list(box3d[i, :3]),
            list(box3d[i, 3:6]),
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


class Box:
    """表示一个 3D 框的数据类，包含标签、分数与速度。"""

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        Args:
            center: 框中心，形如 x, y, z。
            size: 框尺寸，形如 width, length, height。
            orientation: 框朝向（四元数）。
            label: 整数类别标签，可选。
            score: 分类置信度，可选。
            velocity: box 在 x, y, z 方向的速度。
            name: 框名称，可用于表示类别名，可选。
            token: 来自数据库的唯一字符串标识。
        """
        # print(center.shape)
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = (np.allclose(self.velocity, other.velocity) or
               (np.all(np.isnan(self.velocity)) and np.all(np.isnan(other.velocity))))

        return center and wlh and orientation and label and score and vel

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'vel: {:.2f}, {:.2f}, {:.2f}, name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.velocity[0], self.velocity[1], self.velocity[2], self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (3, 3) 的旋转矩阵。
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        平移操作。

        Args:
            x: 形状 (3, 1)，沿 x, y, z 方向的平移量。
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        旋转操作。

        Args:
            quaternion: 要施加的旋转。
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        返回边界框的 8 个角点。

        Args:
            wlh_factor: 将 w、l、h 乘以该系数以缩放框。

        Returns:
            np.ndarray: 形状 (3, 8)。前四个角点朝前，后四个角点朝后。
        """
        w, l, h = self.wlh * wlh_factor

        # 3D 框角点在局部坐标下的定义（约定：x 向前、y 向左、z 向上）
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # 旋转
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # 平移
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        返回底部的四个角点。

        Returns:
            np.ndarray: 形状 (3, 4)。前两个朝前，后两个朝后。
        """
        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        在 matplotlib 坐标轴上绘制该框。

        Args:
            axis: 需要绘制框的坐标轴。
            view: 形状 (3, 3) 的投影矩阵（如做图像内投影时使用）。
            normalize: 是否对剩余坐标归一化。
            colors: 三个 matplotlib 颜色（str 或归一化 RGB 元组），
                分别表示前面、后面与侧面。
            linewidth: 框边线的像素宽度。
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # 绘制四条侧面棱
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # 绘制前面（前 4 角）与后面（后 4 角）的矩形（3D）/ 边线（2D）
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # 绘制指示朝向的线段
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        使用 OpenCV 绘制该框。

        Args:
            im: 形状 (width, height, 3) 的图像数组，通道为 BGR 顺序。
            view: 形状 (3, 3) 的投影矩阵（如做图像内投影时使用）。
            normalize: 是否对剩余坐标归一化。
            colors: 三个 (R, G, B) 颜色，分别表示前面、侧面与后面。
            linewidth: 线宽。
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # 绘制四条侧面棱
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # 绘制前面（前 4 角）与后面（后 4 角）的矩形（3D）/ 边线（2D）
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # 绘制指示朝向的线段
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    def copy(self) -> 'Box':
        """
        返回自身的深拷贝。
        """
        return copy.deepcopy(self)


def visual(points, gt_anno, det, i, eval_range=35, conf_th=0.5):
    """将单帧点云、真值框与预测框绘制为 BEV 图并保存。

    Args:
        points (np.ndarray): 形状 (3+, N) 的点云，前两维为 BEV 坐标。
        gt_anno (dict): 真值框（detection 结构）。
        det (dict): 预测框（detection 结构）。
        i (int): 帧序号，用于生成文件名 demo/file%02d.png。
        eval_range (int): BEV 可视化的坐标范围半径。
        conf_th (float): 预测框显示的置信度阈值。
    """
    _, ax = plt.subplots(1, 1, figsize=(9, 9), dpi=200)
    points = remove_close(points, radius=3)
    points = view_points(points[:3, :], np.eye(4), normalize=False)

    # 用点到原点的距离来映射颜色，越远越亮
    dists = np.sqrt(np.sum(points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / eval_range)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    boxes_gt = _second_det_to_nusc_box(gt_anno)
    boxes_est = _second_det_to_nusc_box(det)

    # 绘制真值框（红色）
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=2)

    # 绘制预测框（蓝色），仅显示高于阈值的框
    for box in boxes_est:
        if box.score >= conf_th:
            box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)


    axes_limit = eval_range + 3  # 略大于评估范围，容纳超出范围的框
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)
    plt.axis('off')

    plt.savefig("demo/file%02d.png" % i)
    plt.close()


def remove_close(points, radius: float) -> None:
    """
    移除距原点一定半径内过近的点。

    Args:
        points (np.ndarray): 形状 (3, N) 的点云。
        radius (float): 需要移除点的半径阈值。
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points