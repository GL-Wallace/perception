"""旋转框相关的纯几何原语。

提供计算几何的基础工具，主要用 numba 加速：凸多边形内点判断(2D/3D)、
线段相交判断与交点求解、3D 平面方程求解等。这些原语是 box_np_ops 中
旋转框 IoU、体素内点聚集、视锥裁剪(spconv 之外用 CPU 几何判断)的底层依赖。

主要函数：
    - points_in_convex_polygon_(3d_)jit: 判断点是否落在凸多边形(2D/3D)内。
    - points_count_convex_polygon_3d_jit: 统计落在 3D 凸多边形内的点数。
    - is_line_segment_intersection_jit / line_segment_intersection: 线段相交。
    - surface_equ_3d(_jitv2): 由表面三点求解平面方程 ax+by+cz+d=0 的 (a,b,c) 与 d。

设计思路：把旋转框视为多个面/线段围成的凸区域，内点判断通过「点在所有
面向内的半空间内」这一判据实现；3D 面的法向量约定指向框内部，因此点在
框内等价于对所有面均有 sign >= 0(或根据具体函数约定取反)。

注意:
    文件末尾保留了几个未在项目中被调用的旧版变体(如 surface_equ_3d_jit、
    points_in_convex_polygon_3d_jit_v1、points_in_convex_polygon_3d_jit_v2、
    _points_in_convex_polygon_3d_jit_v2)，其中部分函数引用了未定义的变量，
    属于历史遗留死代码，不参与实际运行路径。
"""
import numba
import numpy as np


@numba.njit
def _points_count_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces=None
):
    """统计落在多个 3D 凸多边形内的点数(一次性返回每个多边形的计数)。

    Args:
        points: [num_points, 3] 数组。
        polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3]
            数组，所有面的法向量需指向内部，每个面至少 3 个点。
        normal_vec: [num_polygon, max_num_surfaces, 3] 各面的法向量。
        d: [num_polygon, max_num_surfaces] 平面方程常数项。
        num_surfaces: [num_polygon] 数组，指示每个多边形实际包含的面数。

    Returns:
        [num_polygon] 数组，每个元素为落在对应多边形内的点数。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    # 初值设为总点数，一旦某点落在某个面外部就减一
    ret = np.full((num_polygons,), num_points, dtype=np.int64)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                # 代入平面方程，normal 指向内部，故 sign<0 表示点在面外
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[j] -= 1
                    break
    return ret


def points_count_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """统计落在 3D 凸多边形内的点数(包装函数，先求平面方程再调用 jit 内核)。

    Args:
        points: [num_points, 3] 数组。
        polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3]
            数组，所有面的法向量需指向内部，每个面至少 3 个点。
        num_surfaces: [num_polygon] 数组，指示每个多边形的面数。

    Returns:
        [num_polygon] 数组，每个元素为落在对应多边形内的点数。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_count_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def is_line_segment_intersection_jit(lines1, lines2):
    """判断两组线段两两之间是否相交(基于叉积符号的跨立实验)。

    Args:
        lines1 (float, [N, 2, 2]): 第一组线段，每条为 [起点, 终点]。
        lines2 (float, [M, 2, 2]): 第二组线段。

    Returns:
        [N, M] bool 数组，ret[i, j] 为 True 表示 lines1[i] 与 lines2[j] 相交。
    """
    # Return true if line segments AB and CD intersect
    # 经典跨立判据：AB 与 CD 相交当且仅当 C、D 在 AB 两侧且 A、B 在 CD 两侧
    N = lines1.shape[0]
    M = lines2.shape[0]
    ret = np.zeros((N, M), dtype=np.bool_)
    for i in range(N):
        for j in range(M):
            A = lines1[i, 0]
            B = lines1[i, 1]
            C = lines2[j, 0]
            D = lines2[j, 1]
            # 叉积符号判断点 C/D 相对 AB 的方位是否相反
            acd = (D[1] - A[1]) * (C[0] - A[0]) > (C[1] - A[1]) * (D[0] - A[0])
            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
            if acd != bcd:
                abc = (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
                abd = (D[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (D[0] - A[0])
                if abc != abd:
                    ret[i, j] = True
    return ret


@numba.njit
def line_segment_intersection(line1, line2, intersection):
    """求解两条线段的交点(若相交则把坐标写入 intersection 并返回 True)。

    Args:
        line1 ([2, 2]): 线段 1 的两个端点。
        line2 ([2, 2]): 线段 2 的两个端点。
        intersection ([2]): 输出数组，用于接收交点坐标。

    Returns:
        bool: 两线段是否相交；相交时 intersection 被写入交点坐标。
    """
    A = line1[0]
    B = line1[1]
    C = line2[0]
    D = line2[1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    # 跨立实验判断两线段是否相交
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            # 用两条线段的参数方程求交点(分母为叉积)
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            intersection[0] = (ABBA * DC0 - BA0 * CDDC) / DH
            intersection[1] = (ABBA * DC1 - BA1 * CDDC) / DH
            return True
    return False


def _ccw(A, B, C):
    """判断 A->B->C 是否构成逆时针方向(叉积为正)。

    Args:
        A, B, C: 三组 2D 点(支持广播)。

    Returns:
        布尔数组: A、B、C 逆时针时为 True。
    """
    return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (
        B[..., 1] - A[..., 1]
    ) * (C[..., 0] - A[..., 0])


def is_line_segment_cross(lines1, lines2):
    """向量化判断两组线段两两是否相交(非 jit 版本，比 jit 慢约 10 倍)。

    Args:
        lines1: [N, 2, 2]。
        lines2: [M, 2, 2]。

    Returns:
        [N, M] bool 数组。
    """
    # 10x slower than jit version with 1000-1000 random lines input.
    # lines1, [N, 2, 2]
    # lines2, [M, 2, 2]
    A = lines1[:, 0, :][:, np.newaxis, :]
    B = lines1[:, 1, :][:, np.newaxis, :]
    C = lines2[:, 0, :][np.newaxis, :, :]
    D = lines2[:, 1, :][np.newaxis, :, :]
    # 与 jit 版本相同的跨立判据，用广播矩阵表达
    return np.logical_and(
        _ccw(A, C, D) != _ccw(B, C, D), _ccw(A, B, C) != _ccw(A, B, D)
    )


@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    """由表面点求 3D 平面的法向量与常数项(旧版，未使用且含未定义变量)。

    本函数是 surface_equ_3d_jitv2 的历史版本，函数体末尾返回了未定义的
    normal_vec, 实际不可调用；项目当前使用 surface_equ_3d_jitv2。
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_v = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # 取前两个边向量的叉积作为法向量
    # normal_vec: [..., 3]
    normal_v = np.cross(surface_v[:, :, 0, :], surface_v[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    # 用第一个点求 d：d = -(n . p0)
    d = np.einsum("aij, aij->ai", normal_v, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit_v1(points, polygon_surfaces, num_surfaces=None):
    """判断点是否落在 3D 凸多边形内(旧版，调用未定义的 surface_equ_3d_jit)。

    本函数依赖 surface_equ_3d_jit 返回的值，由于该函数含未定义变量，
    实际不可调用；项目当前使用 points_in_convex_polygon_3d_jit。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def surface_equ_3d(polygon_surfaces):
    """由表面点求 3D 平面的法向量与常数项(向量化 NumPy 版)。

    Args:
        polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]。

    Returns:
        (normal_vec, -d): normal_vec 形状 [..., 3] 为法向量，-d 为平面方程常数项。
    """
    # return [a, b, c], d in ax+by+cz+d=0
    surface_v = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    normal_v = np.cross(surface_v[:, :, 0, :], surface_v[:, :, 1, :])
    d = np.einsum("aij, aij->ai", normal_v, polygon_surfaces[:, :, 0, :])
    return normal_v, -d


def points_in_convex_polygon_3d_jit(points, polygon_surfaces, num_surfaces=None):
    """判断点是否落在 3D 凸多边形内。

    Args:
        points: [num_points, 3] 数组。
        polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3]
            数组，所有面的法向量需指向内部，每个面至少 3 个点。
        num_surfaces: [num_polygon] 数组，指示每个多边形的面数。

    Returns:
        [num_points, num_polygon] bool 数组。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, num_surfaces
    )


@numba.njit
def _points_in_convex_polygon_3d_jit(
    points, polygon_surfaces, normal_vec, d, num_surfaces=None
):
    """判断点是否落在 3D 凸多边形内(jit 内核)。

    Args:
        points: [num_points, 3] 数组。
        polygon_surfaces: [num_polygon, max_num_surfaces, max_num_points_of_surface, 3]。
        normal_vec: [num_polygon, max_num_surfaces, 3] 各面法向量(指向内部)。
        d: [num_polygon, max_num_surfaces] 平面方程常数项。
        num_surfaces: [num_polygon] 数组。

    Returns:
        [num_points, num_polygon] bool 数组。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    # 初始置 True，一旦某面判定在外部(法向量的反向半空间)即判为 False
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.jit
def points_in_convex_polygon_jit(points, polygon, clockwise=True):
    """判断点是否落在 2D 凸多边形内(逐点逐面叉积)。

    Args:
        points: [num_points, 2] 数组。
        polygon: [num_polygon, num_points_of_polygon, 2] 数组。
        clockwise: bool，指示多边形顶点是否为顺时针。

    Returns:
        [num_points, num_polygon] bool 数组。
    """
    # first convert polygon to directed lines
    # 把多边形转为有向边向量，用于后续与点到顶点的叉积判断
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = (
            polygon
            - polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
        )
    else:
        vec1 = (
            polygon[
                :,
                [num_points_of_polygon - 1] + list(range(num_points_of_polygon - 1)),
                :,
            ]
            - polygon
        )
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                # 边向量叉乘(顶点到点)向量，符号一致则点在多边形内
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def points_in_convex_polygon(points, polygon, clockwise=True):
    """判断点是否落在凸多边形内(向量化版，比逐点版更快)。

    Args:
        points: [num_points, 2] 数组。
        polygon: [num_polygon, num_points_of_polygon, 2] 数组。
        clockwise: bool，指示顶点顺序。

    Returns:
        [num_points, num_polygon] bool 数组。
    """
    # first convert polygon to directed lines
    num_lines = polygon.shape[1]
    polygon_next = polygon[:, [num_lines - 1] + list(range(num_lines - 1)), :]
    if clockwise:
        vec1 = (polygon - polygon_next)[np.newaxis, ...]
    else:
        vec1 = (polygon_next - polygon)[np.newaxis, ...]
    vec2 = polygon[np.newaxis, ...] - points[:, np.newaxis, np.newaxis, :]
    # [num_points, num_polygon, num_points_of_polygon, 2]
    # 对所有边求叉积，全部大于零表示点在所有边同侧(内部)
    cross = np.cross(vec1, vec2)
    return np.all(cross > 0, axis=2)


@numba.njit
def surface_equ_3d_jitv2(surfaces):
    """由表面点求 3D 平面法向量与常数项(numab jit 内核版本)。

    Args:
        surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]，
            每个面仅使用前 3 个点。

    Returns:
        (normal_vec, d): normal_vec [num_polygon, max_num_surfaces, 3] 为法向量，
            d 为平面方程 ax+by+cz+d=0 的常数项。
    """
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    # 用前 3 个点构造两个边向量，逐步改写 sv0/sv1 以避免小数组分配
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            # 法向量 = sv0 x sv1
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            # d = -(n . p0)
            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )
    return normal_vec, d


@numba.njit
def _points_in_convex_polygon_3d_jit_v2(points, surfaces):
    """3D 凸多边形内点判断(旧版内核，含未定义变量，不可调用)。

    本函数体引用了未作为参数传入的 polygon_surfaces、num_surfaces、
    normal_vec、d 等变量，属于历史遗留死代码，未被项目使用。
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


@numba.njit
def points_in_convex_polygon_3d_jit_v2(points, surfaces, num_surfaces=None):
    """判断点是否落在 3D 凸多边形内(内联平面方程求解的内核变体，未使用)。

    与 points_in_convex_polygon_3d_jit 功能等价，但把表面方程求解与内点
    判断合并在单个 jit 函数中；参数 num_surfaces 未实际参与判断。
    """
    num_polygon = surfaces.shape[0]
    max_num_surfaces = surfaces.shape[1]
    num_points = points.shape[0]
    normal_vec = np.zeros((num_polygon, max_num_surfaces, 3), dtype=surfaces.dtype)
    d = np.zeros((num_polygon, max_num_surfaces), dtype=surfaces.dtype)
    sv0 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    sv1 = surfaces[0, 0, 0] - surfaces[0, 0, 1]
    ret = np.ones((num_points, num_polygon), dtype=np.bool_)
    for i in range(num_polygon):
        for j in range(max_num_surfaces):
            sv0[0] = surfaces[i, j, 0, 0] - surfaces[i, j, 1, 0]
            sv0[1] = surfaces[i, j, 0, 1] - surfaces[i, j, 1, 1]
            sv0[2] = surfaces[i, j, 0, 2] - surfaces[i, j, 1, 2]
            sv1[0] = surfaces[i, j, 1, 0] - surfaces[i, j, 2, 0]
            sv1[1] = surfaces[i, j, 1, 1] - surfaces[i, j, 2, 1]
            sv1[2] = surfaces[i, j, 1, 2] - surfaces[i, j, 2, 2]
            normal_vec[i, j, 0] = sv0[1] * sv1[2] - sv0[2] * sv1[1]
            normal_vec[i, j, 1] = sv0[2] * sv1[0] - sv0[0] * sv1[2]
            normal_vec[i, j, 2] = sv0[0] * sv1[1] - sv0[1] * sv1[0]

            d[i, j] = (
                -surfaces[i, j, 0, 0] * normal_vec[i, j, 0]
                - surfaces[i, j, 0, 1] * normal_vec[i, j, 1]
                - surfaces[i, j, 0, 2] * normal_vec[i, j, 2]
            )

    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygon):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret