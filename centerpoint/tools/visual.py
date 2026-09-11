"""3D 检测结果 open3d 可视化工具。

读取推理导出的 visualization.pkl（每项含点云 points 与检测结果 detections），
用 open3d 逐帧展示点云与按类别着色的 3D 框。

主要函数：
    - label2color: 类别序号到 RGB 颜色的映射。
    - corners_to_lines: 将 8 个角点转换为 open3d LineSet 线框。
    - plot_boxes: 将检测框（中心/尺寸/朝向）转换为线框几何列表。

命令行参数：
    --path    可视化文件路径（pickle）。
    --thresh  3D 框显示的置信度阈值。

依赖 det3d.core.bbox.box_np_ops.center_to_corner_box3d 将 center/size/yaw
转换为 3D 框角点。
"""
from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import open3d as o3d
import argparse
import pickle 

def label2color(label):
    """返回类别序号对应的显示颜色。

    Args:
        label (int): 类别序号，范围为 0-3。

    Returns:
        list: 归一化后的 RGB 颜色值。
    """
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255]]

    return colors[label]

def corners_to_lines(qs, color=[204/255, 0, 0]):
    """将 3D 框的 8 个角点转换为 open3d 线框几何。

    Args:
        qs (np.ndarray): 形状 (8, 3) 的顶点坐标，顶点顺序如下：

                7 -------- 4
               /|         /|
              6 -------- 5 .
              | |        | |
              . 3 -------- 0
              |/         |/
              2 -------- 1

        color (list): 线框颜色（RGB）。

    Returns:
        open3d.geometry.LineSet: 由 12 条边构成的 3D 框线框。
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)
    
    return line_set

def plot_boxes(boxes, score_thresh):
    """将检测结果转换为 open3d 线框列表。

    Args:
        boxes (dict): 检测结果，需含 'scores'、'boxes'（center/size/yaw）、
            'classes' 字段。
        score_thresh (float): 分数线，低于此分数的框会被过滤。

    Returns:
        list: 过滤后各框对应的 LineSet 几何列表。
    """
    visuals =[] 
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue 

        box = boxes['boxes'][i:i+1]
        label = boxes['classes'][i]
        # 将中心/尺寸/朝向转换为 8 个角点坐标
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    # 读取 visualization.pkl：每一项含点云与对应检测结果
    with open(args.path, 'rb') as f:
        data_dicts = pickle.load(f)

    for data in data_dicts:
        points = data['points']
        detections = data['detections']

        pcd = o3d.geometry.PointCloud()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        visual = [pcd]
        num_dets = detections['scores'].shape[0]
        visual += plot_boxes(detections, args.thresh)

        o3d.visualization.draw_geometries(visual)
