"""Waymo 跟踪评估的主入口脚本。

读取第一阶段检测结果（pickle），将每帧 box 从车辆坐标系变换到全局坐标系，
逐帧调用 PubTracker.step_centertrack 得到跟踪 ID，并以 Waymo 官方评测所需的
格式输出 tracking_pred.bin 文件。

主要函数：
    - main: 逐帧跟踪并输出评测文件。
    - convert_detection_to_global_box: 把检测 box 变换到全局坐标系并按序列排序。
    - transform_box: 单帧 box 的中心/朝向/速度变换。
    - sort_detections: 按 (序列 id, 帧 id) 排序。
    - veh_pos_to_transform: 车辆位姿矩阵 <-> 变换矩阵。

与其他模块关系：
    - 依赖 tracker.PubTracker 做跟踪；
    - 依赖 det3d.datasets.waymo.waymo_common._create_pd_detection 输出评测文件。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import copy
import argparse
import copy
import json
import os
import numpy as np
from tools.waymo_tracking.tracker import PubTracker as Tracker
from tqdm import tqdm
import json 
import time
from nuscenes.utils.geometry_utils import transform_matrix
import pickle 
from pyquaternion import Quaternion
from det3d.datasets.waymo.waymo_common import _create_pd_detection

def parse_args():
    """解析命令行参数（工作目录、预测文件、info 路径、各类别距离阈值等）。

    Returns:
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to prediction file"
    )
    parser.add_argument(
        "--info_path", type=str
    )
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--vehicle", type=float, default=0.8) 
    parser.add_argument("--pedestrian", type=float, default=0.4)  
    parser.add_argument("--cyclist", type=float, default=0.6)  
    parser.add_argument("--score_thresh", type=float, default=0.75)

    args = parser.parse_args()

    return args

def get_obj(path):
    """读取 pickle 文件为 Python 对象。"""
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def veh_pos_to_transform(veh_pos):
    """把 4x4 车辆位姿矩阵转换为全局<->车辆两个变换矩阵。

    Args:
        veh_pos (ndarray): (4, 4) 位姿矩阵，左上 3x3 为旋转，右上 3x1 为平移。

    Returns:
        (ndarray, ndarray): 车辆到全局、全局到车辆的 4x4 变换矩阵。
    """
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

def reorganize_info(infos):
    """把 info 列表重组为以 token 为键的字典，方便按帧查询。"""
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

def main():
    """逐帧执行 Waymo 跟踪，输出官方评测所需的 tracking_pred.bin。"""
    args = parse_args()
    print('Deploy OK')

    # 各类别的最近点匹配距离上限（命令行参数，需按模型调优）。
    max_dist = {
        'VEHICLE': args.vehicle,
        'PEDESTRIAN': args.pedestrian,
        'CYCLIST': args.cyclist
    }

    tracker = Tracker(max_age=args.max_age, max_dist=max_dist, score_thresh=args.score_thresh)

    with open(args.checkpoint, 'rb') as f:
        predictions=pickle.load(f)

    with open(args.info_path, 'rb') as f:
        infos=pickle.load(f)
        infos = reorganize_info(infos)

    global_preds, detection_results = convert_detection_to_global_box(predictions, infos)
    size = len(global_preds)

    print("Begin Tracking {} frames\n".format(size))

    predictions = {} 

    for i in tqdm(range(size)):
        pred = global_preds[i]
        token = pred['token']

        # 每个新视频序列（frame_id 回到 0）开始时重置跟踪器。
        if pred['frame_id'] == 0:
            tracker.reset()
            last_time_stamp = pred['timestamp']

        # 距上一帧的时间间隔（秒），供速度平移使用。
        time_lag = (pred['timestamp'] - last_time_stamp) 
        last_time_stamp = pred['timestamp']

        current_det = pred['global_boxs']

        outputs = tracker.step_centertrack(current_det, time_lag)
        tracking_ids = []
        box_ids = [] 

        # 只记录 active 轨迹；丢失但仍保留的轨迹不输出到当前帧。
        for item in outputs:
            if item['active'] == 0:
                continue 
            
            box_ids.append(item['box_id'])
            tracking_ids.append(item['tracking_id'])

        # 用跟踪器保留的 box_id 对原始检测结果做索引重排。
        detection = detection_results[token]

        remained_box_ids = np.array(box_ids)

        track_result = {} 

        # 保存跟踪 ID。
        track_result['tracking_ids']= np.array(tracking_ids)   

        # 保存 box 参数（按保留的 box_id 索引）。
        track_result['box3d_lidar'] = detection['box3d_lidar'][remained_box_ids]

        # 保存 box 类别。
        track_result['label_preds'] = detection['label_preds'][remained_box_ids]

        # 保存 box 分数。
        track_result['scores'] = detection['scores'][remained_box_ids]

        predictions[token] = track_result 

    os.makedirs(args.work_dir, exist_ok=True)
    # 输出 Waymo 官方评测格式的预测文件到 work_dir。
    _create_pd_detection(predictions, infos, args.work_dir, tracking=True)

    result_path = os.path.join(args.work_dir, 'tracking_pred.bin')
    gt_path = os.path.join(args.work_dir, '../gt_preds.bin')

    print("Use Waymo devkit or online server to evaluate the result")
    print("After building the devkit, you can use the following command")
    print("waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main \
           {}  {} ".format(result_path, gt_path))
    
    # os.system("waymo_open_dataset/metrics/tools/compute_tracking_metrics_main \
    #       {}  {} ".format(result_path, gt_path))

def transform_box(box, pose):
    """用 4x4 变换矩阵变换 box 的中心、朝向与速度。

    box 采用 [x, y, z, w, l, h, vx, vy, yaw] 编码（Waymo 检测输出格式）。

    Args:
        box (ndarray): (N, 9) 的 box。
        pose (ndarray): (4, 4) 变换矩阵（这里传入车辆到全局的变换）。

    Returns:
        ndarray: (N, 9) 变换后的 box。
    """
    transform = pose 
    # 朝向：原朝向角加上旋转矩阵的 yaw 分量（arctan2(R[1,0], R[0,0])）。
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0])
    # 中心：旋转后加平移。
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2)

    velocity = box[..., [6, 7]] 

    # 速度补一维 z=0 后与旋转矩阵相乘，再丢弃 z 维，得到旋转后的 2D 速度。
    velocity = np.concatenate([velocity, np.zeros((velocity.shape[0], 1))], axis=-1)

    velocity = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    velocity)[..., [0, 1]]

    return np.concatenate([center, box[..., 3:6], velocity, heading[..., np.newaxis]], axis=-1)

def label_to_name(label):
    """把类别索引映射为 Waymo 类别名。"""
    if label == 0:
        return "VEHICLE"
    elif label == 1 :
        return "PEDESTRIAN"
    elif label == 2:
        return "CYCLIST"
    else:
        raise NotImplemented()

def sort_detections(detections):
    """按 (序列 id, 帧 id) 对帧排序，保证跟踪按时间顺序处理。

    token 形如 "segment_xxxx_..._frameid" 形式，从中解析序列与帧编号。
    """
    indices = [] 

    for det in detections:
        f = det['token']
        # 从 token 字符串解析序列 id 与帧 id。
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        # 组合成一个可排序的整数键：先按序列、再按帧。
        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    detections = [detections[r] for r in rank]

    return detections

def convert_detection_to_global_box(detections, infos):
    """把每帧检测 box 变换到全局坐标系，并按序列/帧排序返回。

    同时返回一份深拷贝的原始检测结果，供后续按 box_id 还原 box 参数与分数。

    Args:
        detections (dict): token -> 检测结果（box3d_lidar/label_preds/scores）。
        infos (dict): token -> 帧 info（含 anno_path/timestamp）。

    Returns:
        (list[dict], dict): 排序后的全局帧列表与原始检测结果字典。
    """
    ret_list = [] 

    detection_results = {} 

    for token in tqdm(infos.keys()):
        detection = detections[token]
        detection_results[token] = copy.deepcopy(detection)

        info = infos[token]
        anno_path = info['anno_path']
        ref_obj = get_obj(anno_path)
        # 车辆到全局的 4x4 变换矩阵。
        pose = np.reshape(ref_obj['veh_to_global'], [4, 4])

        box3d = detection["box3d_lidar"].detach().clone().cpu().numpy() 
        labels = detection["label_preds"].detach().clone().cpu().numpy()
        scores = detection['scores'].detach().clone().cpu().numpy()
        # 朝向角符号取反并旋转 -pi/2，从 CenterPoint 的角约定转换到 Waymo 约定。
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        # 交换宽高维度（w <-> l），适配 Waymo 的 (width, length) 顺序。
        box3d[:, [3, 4]] = box3d[:, [4, 3]]

        # 变换到全局坐标系。
        box3d = transform_box(box3d, pose)

        frame_id = token.split('_')[3][:-4]

        num_box = len(box3d)

        anno_list =[]
        for i in range(num_box):
            # 组装跟踪器所需的逐框字段。
            anno = {
                'translation': box3d[i, :3],
                'velocity': box3d[i, [6, 7]],
                'detection_name': label_to_name(labels[i]),
                'score': scores[i], 
                'box_id': i 
            }

            anno_list.append(anno)

        ret_list.append({
            'token': token, 
            'frame_id':int(frame_id),
            'global_boxs': anno_list,
            'timestamp': info['timestamp'] 
        })

    sorted_ret_list = sort_detections(ret_list)

    return sorted_ret_list, detection_results 

if __name__ == '__main__':
    main()
