"""NumPy 版旋转框与点云几何操作。

提供点云数据处理所依赖的 NumPy/numba 工具集：旋转框角点生成、2D/3D IoU、
包围盒与角点互转、相机/雷达坐标变换、视锥内的点裁剪、体素标签赋值以及
点云增广中会用到的旋转/缩放辅助函数。多数热点函数使用 numba nopython 加速。

主要函数：
    - iou_jit / iou_3d_jit / iou_nd_jit: 轴对齐框的 2D/3D/ND IoU。
    - center_to_corner_box3d / center_to_corner_box2d: 中心+尺寸+角度转角点。
    - corner_to_surfaces_3d(_jit): 角点转面向内的 3D 表面。
    - points_in_rbbox / points_count_rbbox: 判断/统计框内点。
    - assign_label_to_voxel(_v3): 给体素赋 0/1 标签。
    - camera_to_lidar / lidar_to_camera 及 box 版本: 相机与雷达互转。

依赖 det3d.core.bbox.geometry 的内点判断与平面方程原语；被 sampler、
datasets 中的数据流水线以及后处理模块调用。
"""
from pathlib import Path

import numba
import numpy as np
from det3d.core.bbox.geometry import (
    points_count_convex_polygon_3d_jit,
    points_in_convex_polygon_3d_jit,
)
try:
    from spconv.utils import rbbox_intersection, rbbox_iou
except:
    print("Import spconv fail, no support for sparse convolution!")


def points_count_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """统计落在每个旋转框 3D 区域内的点数。

    Args:
        points (np.ndarray): [M, >=3] 点云(仅用前 3 列)。
        rbbox (np.ndarray): [N, 7] 旋转框 [x, y, z, dx, dy, dz, yaw]。
        z_axis (int): 旋转轴，雷达默认绕 z 轴。
        origin (tuple): 原点比例。

    Returns:
        np.ndarray: [N] 每个框包含的点数。
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return points_count_convex_polygon_3d_jit(points[:, :3], surfaces)


def riou_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    """计算两组旋转框在 BEV 平面上的 IoU(调用 spconv 的 CPU 实现)。

    Args:
        rbboxes (np.ndarray): [N, 5] 旋转框 [x, y, dx, dy, yaw]。
        qrbboxes (np.ndarray): [M, 5] 旋转框。
        standup_thresh (float): 外接框 IoU 低于该值则跳过精算。

    Returns:
        np.ndarray: [N, M] IoU 矩阵。
    """
    # less than 50ms when used in second one thread. 10x slower than gpu
    # 先转 BEV 角点与外接框，用粗筛 IoU 加速后续精确 IoU 计算
    boxes_corners = center_to_corner_box2d(
        rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
    )
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(
        qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
    )
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    # if standup box not overlapped, rbbox not overlapped too.
    # 外接框不重叠则旋转框必不重叠
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return rbbox_iou(boxes_corners, qboxes_corners, standup_iou, standup_thresh)


def rinter_cc(rbboxes, qrbboxes, standup_thresh=0.0):
    """计算两组旋转框在 BEV 平面上的相交面积(调用 spconv 的 CPU 实现)。

    Args:
        rbboxes (np.ndarray): [N, 5] 旋转框。
        qrbboxes (np.ndarray): [M, 5] 旋转框。
        standup_thresh (float): 外接框 IoU 低于该值则跳过精算。

    Returns:
        np.ndarray: [N, M] 相交面积矩阵。
    """
    # less than 50ms when used in second one thread. 10x slower than gpu
    boxes_corners = center_to_corner_box2d(
        rbboxes[:, :2], rbboxes[:, 2:4], rbboxes[:, 4]
    )
    boxes_standup = corner_to_standup_nd(boxes_corners)
    qboxes_corners = center_to_corner_box2d(
        qrbboxes[:, :2], qrbboxes[:, 2:4], qrbboxes[:, 4]
    )
    qboxes_standup = corner_to_standup_nd(qboxes_corners)
    # if standup box not overlapped, rbbox not overlapped too.
    standup_iou = iou_jit(boxes_standup, qboxes_standup, eps=0.0)
    return rbbox_intersection(
        boxes_corners, qboxes_corners, standup_iou, standup_thresh
    )


def corners_nd(dims, origin=0.5):
    """根据各维尺寸与原点生成相对盒角点。

    Args:
        dims (np.ndarray): 形状 [N, ndim] 的各维尺寸。
        origin (list or array or float): 原点相对最小角点的比例。

    Returns:
        np.ndarray: 形状 [N, 2**ndim, ndim] 的相对角点。
            布局示例(2d): x0y0, x0y1, x1y0, x1y1；(3d) 8 角点为 x0<x1、y0<y1、z0<z1。
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # 现在 corners_norm 的布局为(2d) x0y0, x0y1, x1y0, x1y1，
    # (3d) 8 个角点按二进制索引排列，需重排为便于后续运算的顺序
    if ndim == 2:
        # generate clockwise box corners
        # 2d 框重排为从最小点起顺时针
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


@numba.njit
def corners_2d_jit(dims, origin=0.5):
    """生成 2D 相对角点(jit 内核版，顺时针 x0y0, x0y1, x1y1, x1y0)。

    Args:
        dims (np.ndarray): [N, 2] 尺寸。
        origin (float): 原点比例。

    Returns:
        np.ndarray: [N, 4, 2] 角点。
    """
    ndim = 2
    corners_norm = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=dims.dtype)
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corners_3d_jit(dims, origin=0.5):
    """生成 3D 相对角点(jit 内核版)。

    Args:
        dims (np.ndarray): [N, 3] 尺寸。
        origin (float): 原点比例。

    Returns:
        np.ndarray: [N, 8, 3] 角点。
    """
    ndim = 3
    corners_norm = np.array(
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        dtype=dims.dtype,
    ).reshape((8, 3))
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape((-1, 1, ndim)) * corners_norm.reshape((1, 2 ** ndim, ndim))
    return corners


@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """由角点求轴对齐外接框(jit 内核版)。

    Args:
        boxes_corner (np.ndarray): [N, num_corners, ndim] 角点。

    Returns:
        np.ndarray: [N, 2*ndim]，前 ndim 为各轴最小值，后 ndim 为最大值。
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def corner_to_standup_nd(boxes_corner):
    """由角点求轴对齐外接框(向量化版)。

    Args:
        boxes_corner (np.ndarray): [N, num_corners, ndim] 角点。

    Returns:
        np.ndarray: [N, 2*ndim]，前 ndim 为各轴最小值，后 ndim 为最大值。
    """
    assert len(boxes_corner.shape) == 3
    standup_boxes = []
    standup_boxes.append(np.min(boxes_corner, axis=1))
    standup_boxes.append(np.max(boxes_corner, axis=1))
    return np.concatenate(standup_boxes, -1)


def rbbox2d_to_near_bbox(rbboxes):
    """把旋转框转为最近似的『竖立/躺倒』轴对齐包围框。

    Args:
        rbboxes (np.ndarray): [N, 5(x, y, xdim, ydim, rad)] 旋转框。

    Returns:
        np.ndarray: [N, 4(xmin, ymin, xmax, ymax)] 轴对齐框。

    注意:
        当 |yaw| 归一化后超过 pi/4 时交换长宽维度，从而得到面积最接近的
        轴对齐框。
    """
    rots = rbboxes[..., -1]
    # 把角度折到 [0, pi/2) 内以便判断框更接近竖立还是躺倒
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    # 角度较大时交换 dx/dy 位置
    bboxes_center = np.where(cond, rbboxes[:, [0, 1, 3, 2]], rbboxes[:, :4])
    bboxes = center_to_minmax_2d(bboxes_center[:, :2], bboxes_center[:, 2:])
    return bboxes


def rotation_3d_in_axis(points, angles, axis=0):
    """沿指定坐标轴批量旋转点集(NumPy 版)。

    Args:
        points (np.ndarray): [N, point_size, 3] 待旋转点。
        angles (np.ndarray): [N] 旋转角。
        axis (int): 0(x 轴)、1(y 轴)或 2/-1(z 轴)。

    Returns:
        np.ndarray: 旋转后的点集。
    """
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_points_single_angle(points, angle, axis=0):
    """用单一角度绕轴旋转点集(所有点共享一个角度)。

    Args:
        points (np.ndarray): [N, 3] 待旋转点。
        angle (float): 旋转角。
        axis (int): 旋转轴。

    Returns:
        np.ndarray: 旋转后的点集。
    """
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T


def rotation_2d(points, angles):
    """绕原点旋转 2D 点(角度为正时顺时针)。

    Args:
        points (np.ndarray): [N, point_size, 2] 待旋转点。
        angles (np.ndarray): [N] 旋转角。

    Returns:
        np.ndarray: 与 points 同形状的旋转结果。
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def rotation_box(box_corners, angle):
    """用单一角度旋转一组 2D 框角点。

    Args:
        box_corners (np.ndarray): [N, point_size, 2] 角点。
        angle (float): 旋转角。

    Returns:
        np.ndarray: 旋转后的角点。
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T = np.array(
        [[rot_cos, -rot_sin], [rot_sin, rot_cos]], dtype=box_corners.dtype
    )
    return box_corners @ rot_mat_T


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """由中心、尺寸、角度把 kitti 风格框转为 8 角点。

    Args:
        centers (np.ndarray): [N, 3] 中心坐标。
        dims (np.ndarray): [N, 3] 尺寸。
        angles (np.ndarray): [N] 旋转角，可为 None。
        origin (list or array or float): 原点比例，相机 [0.5, 1.0, 0.5]，
            雷达 [0.5, 0.5, 0]。
        axis (int): 旋转轴，1 为相机、2 为雷达。

    Returns:
        np.ndarray: [N, 8, 3] 角点。
    """
    # kitti 的 length 在 x 轴上；相机/雷达尺寸顺序与 yaw 约定不同，详见源码注释
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """由中心、尺寸、角度把 2D 框转为 4 角点。

    Args:
        centers (np.ndarray): [N, 2] 中心坐标。
        dims (np.ndarray): [N, 2] 尺寸。
        angles (np.ndarray): [N] 旋转角(正为顺时针)，可为 None。
        origin (float): 原点比例。

    Returns:
        np.ndarray: [N, 4, 2] 角点。
    """
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return corners


@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    """把 [x, y, dx, dy, yaw] 框逐条转为 4 角点(jit 版)。

    Args:
        boxes (np.ndarray): [N, 5] 框。

    Returns:
        np.ndarray: [N, 4, 2] 角点。
    """
    num_box = boxes.shape[0]
    # 归一化角点(相对中心)，先构造单位框再减去中心偏移
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
        # 旋转后平移到框中心
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.5], axis=2):
    """把 [x, y, z, dx, dy, dz, yaw] 框批量转为 8 角点。

    Args:
        rbboxes (np.ndarray): [N, 7] 旋转框。
        origin (list): 原点比例。
        axis (int): 旋转轴。

    Returns:
        np.ndarray: [N, 8, 3] 角点。
    """
    return center_to_corner_box3d(
        rbboxes[..., :3], rbboxes[..., 3:6], rbboxes[..., 6], origin, axis=axis
    )


def rbbox3d_to_bev_corners(rbboxes, origin=0.5):
    """把 3D 框投影到 BEV 得到其 4 个底面角点。

    Args:
        rbboxes (np.ndarray): [N, 7] 旋转框。
        origin (float): 原点比例。

    Returns:
        np.ndarray: [N, 4, 2] BEV 角点。
    """
    return center_to_corner_box2d(
        rbboxes[..., :2], rbboxes[..., 3:5], rbboxes[..., 6], origin
    )


def minmax_to_corner_2d(minmax_box):
    """把 min/max 表示转为 2D 角点。

    Args:
        minmax_box (np.ndarray): [..., 4] [xmin, ymin, xmax, ymax]。

    Returns:
        np.ndarray: [..., 4, 2] 角点。
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def minmax_to_corner_2d_v2(minmax_box):
    """把 min/max 表示直接索引重排为 2D 角点(更快)。

    Args:
        minmax_box (np.ndarray): [N, 4] [xmin, ymin, xmax, ymax]。

    Returns:
        np.ndarray: [N, 4, 2] 角点。
    """
    # N, 4 -> N 4 2
    # 手动拼出四个角点：x0y0, x0y1, x1y1, x1y0
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def minmax_to_corner_3d(minmax_box):
    """把 min/max 表示转为 3D 角点。

    Args:
        minmax_box (np.ndarray): [..., 6] [xmin, ymin, zmin, xmax, ymax, zmax]。

    Returns:
        np.ndarray: [..., 8, 3] 角点。
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box3d(center, dims, origin=0.0)


def minmax_to_center_2d(minmax_box):
    """把 min/max 表示转为中心+尺寸表示。

    Args:
        minmax_box (np.ndarray): [..., 4] [xmin, ymin, xmax, ymax]。

    Returns:
        np.ndarray: [..., 4] [cx, cy, dx, dy]。
    """
    ndim = minmax_box.shape[-1] // 2
    center_min = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center_min
    center = center_min + 0.5 * dims
    return np.concatenate([center, dims], axis=-1)


def center_to_minmax_2d_0_5(centers, dims):
    """中心+尺寸(原点 0.5)转为 min/max 表示。

    Args:
        centers (np.ndarray): [..., 2]。
        dims (np.ndarray): [..., 2]。

    Returns:
        np.ndarray: [..., 4] [xmin, ymin, xmax, ymax]。
    """
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)


def center_to_minmax_2d(centers, dims, origin=0.5):
    """中心+尺寸转为 min/max 表示。

    Args:
        centers (np.ndarray): [..., 2]。
        dims (np.ndarray): [..., 2]。
        origin (float): 原点比例。

    Returns:
        np.ndarray: [..., 4] [xmin, ymin, xmax, ymax]。
    """
    if origin == 0.5:
        return center_to_minmax_2d_0_5(centers, dims)
    corners = center_to_corner_box2d(centers, dims, origin=origin)
    return corners[:, [0, 2]].reshape([-1, 4])


def limit_period(val, offset=0.5, period=np.pi):
    """把角度折到 [offset*period - period, offset*period] 范围内。

    Args:
        val (np.ndarray): 输入角度。
        offset (float): 周期内的偏移(0.5 表示对称区间)。
        period (float): 周期。

    Returns:
        np.ndarray: 折到目标区间的角度。
    """
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_CRT_kitti(proj):
    """把 KITTI 投影矩阵 P 分解为内参 C、旋转 R 与平移 T。

    P = C @ [R|T]，其中 C 为上三角内参矩阵；通过先求逆再 QR 分解稳定地
    得到 C 与 R。

    Args:
        proj (np.ndarray): 3x4 投影矩阵。

    Returns:
        tuple: (C, R, T)。
    """
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def get_frustum(bbox_image, C, near_clip=0.001, far_clip=100):
    """由图像框与内参在相机坐标系构造视锥的 8 个角点。

    Args:
        bbox_image (np.ndarray): [4] 图像框 [xmin, ymin, xmax, ymax]。
        C (np.ndarray): 3x3 内参矩阵。
        near_clip (float): 近裁剪面距离。
        far_clip (float): 远裁剪面距离。

    Returns:
        np.ndarray: [8, 3] 视角锥 8 个角点(近/远平面各 4 个)。
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[:, np.newaxis]
    b = bbox_image
    # 图像框的 4 个角(左上、左下、右下、右上)
    box_corners = np.array(
        [[b[0], b[1]], [b[0], b[3]], [b[2], b[3]], [b[2], b[1]]], dtype=C.dtype
    )
    # 依据针孔模型把像素坐标反投影到近/远平面
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=0)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=1)
    return ret_xyz


def get_frustum_v2(bboxes, C, near_clip=0.001, far_clip=100):
    """批量构造多个图像框对应的视锥角点。

    Args:
        bboxes (np.ndarray): [N, 4] 图像框。
        C (np.ndarray): 3x3 内参矩阵。
        near_clip (float): 近裁剪面距离。
        far_clip (float): 远裁剪面距离。

    Returns:
        np.ndarray: [N, 8, 3] 视锥角点。
    """
    fku = C[0, 0]
    fkv = -C[1, 1]
    u0v0 = C[0:2, 2]
    num_box = bboxes.shape[0]
    z_points = np.array([near_clip] * 4 + [far_clip] * 4, dtype=C.dtype)[
        np.newaxis, :, np.newaxis
    ]
    z_points = np.tile(z_points, [num_box, 1, 1])
    box_corners = minmax_to_corner_2d_v2(bboxes)
    near_box_corners = (box_corners - u0v0) / np.array(
        [fku / near_clip, -fkv / near_clip], dtype=C.dtype
    )
    far_box_corners = (box_corners - u0v0) / np.array(
        [fku / far_clip, -fkv / far_clip], dtype=C.dtype
    )
    ret_xy = np.concatenate([near_box_corners, far_box_corners], axis=1)  # [8, 2]
    ret_xyz = np.concatenate([ret_xy, z_points], axis=-1)
    return ret_xyz


@numba.njit
def _add_rgb_to_points_kernel(points_2d, image, points_rgb):
    """把图像像素颜色按投影像素坐标赋给点(jit 内核)。

    对每个投影点取最近邻像素的颜色写入 points_rgb，越界点保持原值。

    Args:
        points_2d (np.ndarray): [N, 2] 投影像素坐标。
        image (np.ndarray): [H, W, C] 源图像。
        points_rgb (np.ndarray): [N, 3] 输出颜色数组。
    """
    num_points = points_2d.shape[0]
    image_h, image_w = image.shape[:2]
    for i in range(num_points):
        img_pos = np.floor(points_2d[i]).astype(np.int32)
        if img_pos[0] >= 0 and img_pos[0] < image_w:
            if img_pos[1] >= 0 and img_pos[1] < image_h:
                points_rgb[i, :] = image[img_pos[1], img_pos[0], :]
                # image[img_pos[1], img_pos[0]] = 0


def add_rgb_to_points(points, image, rect, Trv2c, P2, mean_size=[5, 5]):
    """给雷达点云补充对应的图像颜色特征。

    Args:
        points (np.ndarray): [N, >=3] 雷达点云。
        image (np.ndarray): [H, W, C] 图像。
        rect (np.ndarray): 整流矩阵。
        Trv2c (np.ndarray): 雷达->相机外参。
        P2 (np.ndarray): 投影矩阵。
        mean_size (list): 均值滤波核尺寸(当前未实际使用)。

    Returns:
        np.ndarray: [N, 3] 每个点对应的 RGB 颜色。
    """
    kernel = np.ones(mean_size, np.float32) / np.prod(mean_size)
    # image = cv2.filter2D(image, -1, kernel)
    points_cam = lidar_to_camera(points[:, :3], rect, Trv2c)
    points_2d = project_to_image(points_cam, P2)
    points_rgb = np.zeros([points_cam.shape[0], 3], dtype=points.dtype)
    _add_rgb_to_points_kernel(points_2d, image, points_rgb)
    return points_rgb


def project_to_image(points_3d, proj_mat):
    """把 3D 点投影到图像平面(齐次化 + 透视除法)。

    Args:
        points_3d (np.ndarray): [..., 3] 3D 点。
        proj_mat (np.ndarray): 3x4 投影矩阵。

    Returns:
        np.ndarray: [..., 2] 像素坐标。
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    return point_2d_res


def camera_to_lidar(points, r_rect, velo2cam):
    """相机坐标 -> 雷达坐标(自动补齐齐次坐标)。

    Args:
        points (np.ndarray): [..., 3] 相机坐标点。
        r_rect (np.ndarray): 3x3 整流旋转矩阵。
        velo2cam (np.ndarray): 4x4 雷达->相机外参。

    Returns:
        np.ndarray: [..., 3] 雷达坐标点。
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    """雷达坐标 -> 相机坐标(自动补齐齐次坐标)。

    Args:
        points (np.ndarray): [..., 3] 雷达坐标点。
        r_rect (np.ndarray): 3x3 整流旋转矩阵。
        velo2cam (np.ndarray): 4x4 雷达->相机外参。

    Returns:
        np.ndarray: [..., 3] 相机坐标点。
    """
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    """相机框 -> 雷达框(含尺寸与 yaw 顺序调整)。

    相机框 [x, y, z, l, h, w, r] -> 雷达框 [x, y, z, w, l, h, r]。

    Args:
        data (np.ndarray): [N, 7] 相机框。
        r_rect, velo2cam: 见 camera_to_lidar。

    Returns:
        np.ndarray: [N, 7] 雷达框。
    """
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    """雷达框 -> 相机框(含尺寸与 yaw 顺序调整)。

    雷达框 [x, y, z, w, l, h, r] -> 相机框 [x, y, z, l, h, w, r]。

    Args:
        data (np.ndarray): [N, 7] 雷达框。
        r_rect, velo2cam: 见 lidar_to_camera。

    Returns:
        np.ndarray: [N, 7] 相机框。
    """
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def remove_outside_points(points, rect, Trv2c, P2, image_shape):
    """移除投影到图像视野视锥之外的点。

    先由图像边界构造视锥(相机系)再变换回雷达系，最后用凸多边形内点判断
    筛选出视野内的点。

    Args:
        points (np.ndarray): [N, >=3] 雷达点云。
        rect, Trv2c, P2: 相机内外参与投影矩阵。
        image_shape (tuple): (height, width) 图像尺寸。

    Returns:
        np.ndarray: 位于图像视锥内的点。
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = [0, 0, image_shape[1], image_shape[0]]
    frustum = get_frustum(image_bbox, C)
    # 视锥角点从相机系变换到雷达系
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=1.0):
    """计算轴对齐 2D 框的 IoU(jit 版)。

    Args:
        boxes (np.ndarray): [N, 4] [xmin, ymin, xmax, ymax] 框。
        query_boxes (np.ndarray): [K, 4] 查询框。
        eps (float): 加在边长上的小量以避免除零。

    Returns:
        np.ndarray: [N, K] IoU 矩阵。
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (
            query_boxes[k, 3] - query_boxes[k, 1] + eps
        )
        for n in range(N):
            # 求交集的宽与高
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + eps
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + eps
                )
                if ih > 0:
                    # 并集面积 = 两框面积之和 - 交集面积
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps)
                        * (boxes[n, 3] - boxes[n, 1] + eps)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


@numba.jit(nopython=True)
def iou_3d_jit(boxes, query_boxes, add1=True):
    """计算轴对齐 3D 框的 IoU(jit 版)。

    Args:
        boxes (np.ndarray): [N, 6] [xmin, ymin, zmin, xmax, ymax, zmax]。
        query_boxes (np.ndarray): [K, 6]。
        add1 (bool): 边长是否加 1(体素统计口径)。

    Returns:
        np.ndarray: [N, K] IoU 矩阵。
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    for k in range(K):
        box_area = (
            (query_boxes[k, 3] - query_boxes[k, 0] + add1)
            * (query_boxes[k, 4] - query_boxes[k, 1] + add1)
            * (query_boxes[k, 5] - query_boxes[k, 2] + add1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 3], query_boxes[k, 3])
                - max(boxes[n, 0], query_boxes[k, 0])
                + add1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 4], query_boxes[k, 4])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + add1
                )
                if ih > 0:
                    il = (
                        min(boxes[n, 5], query_boxes[k, 5])
                        - max(boxes[n, 2], query_boxes[k, 2])
                        + add1
                    )
                    if il > 0:
                        ua = float(
                            (boxes[n, 3] - boxes[n, 0] + add1)
                            * (boxes[n, 4] - boxes[n, 1] + add1)
                            * (boxes[n, 5] - boxes[n, 2] + add1)
                            + box_area
                            - iw * ih * il
                        )
                        overlaps[n, k] = iw * ih * il / ua
    return overlaps


@numba.jit(nopython=True)
def iou_nd_jit(boxes, query_boxes, add1=True):
    """计算轴对齐 ND 框的 IoU(jit 版，比 iou_jit 慢约 2 倍)。

    Args:
        boxes (np.ndarray): [N, ndim*2]。
        query_boxes (np.ndarray): [K, ndim*2]。
        add1 (bool): 边长是否加 1。

    Returns:
        np.ndarray: [N, K] IoU 矩阵。
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ndim = boxes.shape[1] // 2
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    side_lengths = np.zeros((ndim,), dtype=boxes.dtype)
    if add1:
        add1 = 1.0
    else:
        add1 = 0.0
    invalid = False
    for k in range(K):
        qbox_area = query_boxes[k, ndim] - query_boxes[k, 0] + add1
        for i in range(1, ndim):
            qbox_area *= query_boxes[k, ndim + i] - query_boxes[k, i] + add1
        for n in range(N):
            invalid = False
            for i in range(ndim):
                side_length = (
                    min(boxes[n, i + ndim], query_boxes[k, i + ndim])
                    - max(boxes[n, i], query_boxes[k, i])
                    + add1
                )
                if side_length <= 0:
                    invalid = True
                    break
                side_lengths[i] = side_length
            if not invalid:
                box_area = boxes[n, ndim] - boxes[n, 0] + add1
                for i in range(1, ndim):
                    box_area *= boxes[n, ndim + i] - boxes[n, i] + add1
                inter = side_lengths[0]
                for i in range(1, ndim):
                    inter *= side_lengths[i]
                # inter = np.prod(side_lengths)
                ua = float(box_area + qbox_area - inter)
                overlaps[n, k] = inter / ua

    return overlaps


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0.5)):
    """判断点是否落在每个旋转框内。

    Args:
        points (np.ndarray): [M, >=3] 点云。
        rbbox (np.ndarray): [N, 7] 旋转框。
        z_axis (int): 旋转轴。
        origin (tuple): 原点比例。

    Returns:
        np.ndarray: [M, N] bool 数组，元素为点是否落在对应框内。
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners):
    """由 3D 框角点构造 6 个面(法向量统一指向内部)。

    Args:
        corners (np.ndarray): [N, 8, 3] 角点，必须由本模块的角点函数生成。

    Returns:
        np.ndarray: [N, 6, 4, 3] 表面顶点。
    """
    # box_corners: [N, 8, 3]，须来自本模块的角点函数以保证顶点顺序一致
    surfaces = np.array(
        [
            [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
            [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
            [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
            [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
            [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
            [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
        ]
    ).transpose([2, 0, 1, 3])
    return surfaces


@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """由 3D 框角点构造 6 个面(jit 版，法向量统一指向内部)。

    Args:
        corners (np.ndarray): [N, 8, 3] 角点。

    Returns:
        np.ndarray: [N, 6, 4, 3] 表面顶点。
    """
    # box_corners: [N, 8, 3]，须来自本模块的角点函数以保证顶点顺序一致
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    # 预定义每个面的 4 个顶点索引
    corner_idxes = np.array(
        [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    ).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces


def assign_label_to_voxel(gt_boxes, coors, voxel_size, coors_range):
    """按体素中心是否落在 GT 框内为每个体素赋 0/1 标签(雷达坐标)。

    Args:
        gt_boxes (np.ndarray): [N, 7] GT 框。
        coors (np.ndarray): [M, 3] 体素坐标(通常为 [z, y, x] 顺序)。
        voxel_size (list): 体素尺寸 [vx, vy, vz]。
        coors_range (list): 点云范围 [xmin, ymin, zmin, xmax, ymax, zmax]。

    Returns:
        np.ndarray: [M] 每体素标签(1 表示中心在框内)。
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    # coors 为 [z, y, x] 顺序，取反转为 [x, y, z] 后换算物理坐标
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_centers = voxel_origins + voxel_size * 0.5
    # GT 框外扩半个体素并相应扩尺寸，使体素中心落在框边界时也算入
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3] - voxel_size * 0.5,
        gt_boxes[:, 3:6] + voxel_size,
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    ret = points_in_convex_polygon_3d_jit(voxel_centers, gt_surfaces)
    return np.any(ret, axis=1).astype(np.int64)


def assign_label_to_voxel_v3(gt_boxes, coors, voxel_size, coors_range):
    """按体素是否与 GT 框有重叠(判角点)为每个体素赋 0/1 标签(雷达坐标)。

    Args:
        gt_boxes (np.ndarray): [N, 7] GT 框。
        coors (np.ndarray): [M, 3] 体素坐标([z, y, x] 顺序)。
        voxel_size (list): 体素尺寸。
        coors_range (list): 点云范围。

    Returns:
        np.ndarray: [M] 每体素标签(1 表示体素任一角点落在框内)。
    """
    voxel_size = np.array(voxel_size, dtype=gt_boxes.dtype)
    coors_range = np.array(coors_range, dtype=gt_boxes.dtype)
    shift = coors_range[:3]
    voxel_origins = coors[:, ::-1] * voxel_size + shift
    voxel_maxes = voxel_origins + voxel_size
    voxel_minmax = np.concatenate([voxel_origins, voxel_maxes], axis=-1)
    voxel_corners = minmax_to_corner_3d(voxel_minmax)
    gt_box_corners = center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2,
    )
    gt_surfaces = corner_to_surfaces_3d(gt_box_corners)
    voxel_corners_flat = voxel_corners.reshape([-1, 3])
    ret = points_in_convex_polygon_3d_jit(voxel_corners_flat, gt_surfaces)
    # 每个体素的 8 个角点是否存在落在框内的
    ret = ret.reshape([-1, 8, ret.shape[-1]])
    return ret.any(-1).any(-1).astype(np.int64)


def image_box_region_area(img_cumsum, bbox):
    """用积分图(面积表)快速求图像框区域内的像素和。

    Args:
        img_cumsum (np.ndarray): [M, H, W] 累积和图像(yx 顺序)。
        bbox (np.ndarray): [N, 4] [xmin, ymin, xmax, ymax]。

    Returns:
        np.ndarray: [N, M] 每个框在每个通道上的区域和。

    说明:
        积分图区域和公式 Iabcd = ID - IB - IC + IA。
    """
    N = bbox.shape[0]
    M = img_cumsum.shape[0]
    ret = np.zeros([N, M], dtype=img_cumsum.dtype)
    ID = img_cumsum[:, bbox[:, 3], bbox[:, 2]]
    IA = img_cumsum[:, bbox[:, 1], bbox[:, 0]]
    IB = img_cumsum[:, bbox[:, 3], bbox[:, 0]]
    IC = img_cumsum[:, bbox[:, 1], bbox[:, 2]]
    ret = ID - IB - IC + IA
    return ret


def get_minimum_bounding_box_bv(points, voxel_size, bound, downsample=8, margin=1.6):
    """求覆盖点云的最小 BEV 包围盒(按 downsample 与体素对齐并外扩 margin)。

    Args:
        points (np.ndarray): [N, >=2] 点云。
        voxel_size (list): [vx, vy] 体素尺寸。
        bound (list): 可用的坐标上界 [xmin, ymin, xmax, ymax]。
        downsample (int): 下采样倍数。
        margin (float): 外扩余量。

    Returns:
        np.ndarray: [4] [xmin, ymin, xmax, ymax]。
    """
    x_vsize = voxel_size[0]
    y_vsize = voxel_size[1]
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    # 将边界对齐到 downsample*voxel_size 的格点
    max_x = np.floor(max_x / (x_vsize * downsample) + 1) * (x_vsize * downsample)
    max_y = np.floor(max_y / (y_vsize * downsample) + 1) * (y_vsize * downsample)
    min_x = np.floor(min_x / (x_vsize * downsample)) * (x_vsize * downsample)
    min_y = np.floor(min_y / (y_vsize * downsample)) * (y_vsize * downsample)
    # 外扩 margin 并裁剪到 bound 内
    max_x = np.minimum(max_x + margin, bound[2])
    max_y = np.minimum(max_y + margin, bound[3])
    min_x = np.maximum(min_x - margin, bound[0])
    min_y = np.maximum(min_y - margin, bound[1])
    return np.array([min_x, min_y, max_x, max_y])
    

def box3d_to_bbox(box3d, rect, Trv2c, P2):
    """把 3D 框投影到图像得到 2D 包围框。

    Args:
        box3d (np.ndarray): [N, 7] 雷达框。
        rect, Trv2c, P2: 相机内外参。

    Returns:
        np.ndarray: [N, 4] 图像框 [xmin, ymin, xmax, ymax]。
    """
    box3d_to_cam = box_lidar_to_camera(box3d, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def change_box3d_center_(box3d, src, dst):
    """在原位调整框中心的原点基准(从 src 原点改为 dst 原点)。

    Args:
        box3d (np.ndarray): [..., 7] 框。
        src (list/array): 原原点比例。
        dst (list/array): 新原点比例。

    Returns:
        None: 原地修改 box3d 的中心。
    """
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    # 原点位置变化引起中心相应平移量 = dims * (dst - src)
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)