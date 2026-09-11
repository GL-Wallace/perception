"""Waymo 数据集的 infos 生成与预测/真值结果转换工具。

主要功能:
    - _fill_infos / create_waymo_infos: 从逐帧 pickle 生成 infos_*.pkl，并完成
      Waymo 到内部的坐标、yaw、长宽约定转换，以及多 sweep 的历史帧对齐信息。
    - _create_pd_detection / _create_gt_detection: 将检测结果/真值写成 Waymo
      官方评测所需的 protobuf 二进制文件（detection_pred.bin / gt_preds.bin）。
    - veh_pos_to_transform / sort_frame 等: 位姿与帧序的辅助工具。

坐标约定: Waymo 原始 box 为 [x,y,z,len,wid,hei,vel_x,vel_y,heading]，其中 heading
为从 x 轴正方向顺时针量取；内部统一约定为 [x,y,z,dx,dy,dz,yaw]（KITTI 风格，
yaw 从 y 轴负方向逆时针量取，且 dx/wid、dy/len 对应交换），转换见 _fill_infos。
"""
import os.path as osp
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm
try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("No Tensorflow")

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}
TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_obj(path):
    """从 pickle 文件读出单个对象（帧点云或帧标注）。

    Args:
        path (str): pickle 文件路径。

    Returns:
        object: 反序列化得到的字典对象。
    """
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

# ignore sign class
# 将内部的类别编号映射回 Waymo 评测类别：0->VEHICLE, 1->PEDESTRIAN, 2->CYCLIST（忽略 SIGN）
LABEL_TO_TYPE = {0: 1, 1:2, 2:4}

import uuid 

class UUIDGeneration():
    """为跟踪 ID 生成稳定唯一字符串的工具。

    Waymo 跟踪评测要求每个目标的 id 为字符串；同一 seed（跟踪 ID）在多次调用间
    返回同一 UUID，保证跨帧目标身份一致。
    """
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()

def _create_pd_detection(detections, infos, result_path, tracking=False):
    """将模型预测结果写为 Waymo 官方评测的 protobuf 二进制文件。

    遍历检测结果字典，把内部的 [x,y,z,dx,dy,dz,yaw] box 逆变换回 Waymo 的
    [x,y,z,len,wid,hei,heading] 约定，并补上 score、类别与（可选的）跟踪 ID，
    最终序列化为 detection_pred.bin 或 tracking_pred.bin。

    Args:
        detections (dict): key 为 frame token，value 含 box3d_lidar/scores/label_preds
            等张量的预测结果。
        infos (dict): token -> info 的映射，用于取 scene_name、帧时间戳等元信息。
        result_path (str): 结果输出目录。
        tracking (bool): 若为 True 则输出跟踪结果并附带对象 ID。
    """
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    objects = metrics_pb2.Objects()

    for token, detection in tqdm(detections.items()):
        info = infos[token]
        obj = get_obj(info['anno_path'])

        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()

        # transform back to Waymo coordinate
        # 内部约定(见 _fill_infos)到 Waymo 原约定的逆变换:
        # x,y,z,w,l,h,r2  ->  x,y,z,l,w,h,r1
        # r2 = -pi/2 - r1  =>  r1 = -pi/2 - r2
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        # 内部顺序 [x,y,z,dx,dy,dz,yaw]，其中 dx/dy 对应 Waymo 的 wid/len，此处交换回 len/wid
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]

        if tracking:
            tracking_ids = detection['tracking_ids']

        for i in range(box3d.shape[0]):
            det  = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            # 帧名形如 "{scene}_{location}_{time_of_day}_{timestamp}"，取最后一段为微秒时间戳
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label] 

            if tracking:
                o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

            objects.objects.append(o)

    # Write objects to a file.
    if tracking:
        path = os.path.join(result_path, 'tracking_pred.bin')
    else:
        path = os.path.join(result_path, 'detection_pred.bin')

    print("results saved to {}".format(path))
    f = open(path, 'wb')
    f.write(objects.SerializeToString())
    f.close()

def _create_gt_detection(infos, tracking=True):
    """生成本地评测用的真值 protobuf 文件（gt_preds.bin）。

    从每帧标注 pickle 中读取 GT box，过滤无点与 UNKNOWN 类别后写成 Waymo
    评测所需的 objects 二进制，便于与预测结果做离线指标对比。

    Args:
        infos (list): 帧 info 列表，每项含 anno_path。
        tracking (bool): 保留参数，标记是否为跟踪模式设定。
    """
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    
    objects = metrics_pb2.Objects()

    for idx in tqdm(range(len(infos))): 
        info = infos[idx]

        obj = get_obj(info['anno_path'])
        annos = obj['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        # 标注 box 为 [x,y,z,len,wid,hei,vel_x,vel_y,heading]，取前 7 维用于评测
        box3d = np.array([ann['box'] for ann in annos])

        if len(box3d) == 0:
            continue 

        names = np.array([TYPE_LIST[ann['label']] for ann in annos])

        box3d = box3d[:, [0, 1, 2, 3, 4, 5, -1]]

        for i in range(box3d.shape[0]):
            if num_points_in_gt[i] == 0:
                continue 
            if names[i] == 'UNKNOWN':
                continue 

            det  = box3d[i]
            score = 1.0
            label = names[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = CAT_NAME_TO_ID[label]
            o.object.num_lidar_points_in_box = num_points_in_gt[i]
            o.object.id = annos[i]['name']

            objects.objects.append(o)
        
    # Write objects to a file.
    f = open(os.path.join(args.result_path, 'gt_preds.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()

def veh_pos_to_transform(veh_pos):
    """由 4x4 位姿矩阵构造车体坐标系与全局坐标系的互变换矩阵。

    Args:
        veh_pos (np.ndarray): [4,4] 刚体变换矩阵，表示 vehicle -> global。

    Returns:
        tuple: (global_from_car, car_from_global) 两个齐次变换矩阵。
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

def _fill_infos(root_path, frames, split='train', nsweeps=1):
    """为每帧构建 info 字典，包含点云路径、标注、时间戳与多 sweep 对齐信息。

    对每个参考帧：
        1. 读取其标注 pickle 得到位姿与 GT box；
        2. 将 Waymo box 转换为内部约定（yaw = -pi/2 - heading，并交换长宽）；
        3. 向上追溯 nsweeps-1 个历史帧，计算把历史点云对齐到当前帧坐标系的
           transform_matrix 与 time_lag，存入 info["sweeps"]。

    Args:
        root_path (str): 数据集根目录（其下含 {split}/lidar 与 {split}/annos）。
        frames (list): 排序后的帧文件名列表。
        split (str): 数据集切分（train/val/test）。
        nsweeps (int): 聚合的 sweep 数量（含当前帧）。

    Returns:
        list: 逐帧 info 字典列表。
    """
    # load all train infos
    infos = []
    for frame_name in tqdm(frames):  # global id
        lidar_path = os.path.join(root_path, split, 'lidar', frame_name)
        ref_path = os.path.join(root_path, split, 'annos', frame_name)

        ref_obj = get_obj(ref_path)
        # 帧名最后一段为微秒时间戳，乘以 1e-6 转为秒
        ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        _, ref_from_global = veh_pos_to_transform(ref_pose)

        info = {
            "path": lidar_path,
            "anno_path": ref_path, 
            "token": frame_name,
            "timestamp": ref_time,
            "sweeps": []
        }

        # 帧名形如 "seq_{n}_{...}_frame_{m}.pkl"，解析出序列号与帧号
        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4]) # remove .pkl

        prev_id = frame_id
        sweeps = [] 
        while len(sweeps) < nsweeps - 1:
            if prev_id <= 0:
                # 序列开头已无更早帧：首个 sweep 退化为当前帧（无变换），后续用同一 sweep 补齐
                if len(sweeps) == 0:
                    sweep = {
                        "path": lidar_path,
                        "token": frame_name,
                        "transform_matrix": None,
                        "time_lag": 0
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_id = prev_id - 1
                # global identifier  

                curr_name = 'seq_{}_frame_{}.pkl'.format(sequence_id, prev_id)
                curr_lidar_path = os.path.join(root_path, split, 'lidar', curr_name)
                curr_label_path = os.path.join(root_path, split, 'annos', curr_name)
                
                curr_obj = get_obj(curr_label_path)
                curr_pose = np.reshape(curr_obj['veh_to_global'], [4, 4])
                global_from_car, _ = veh_pos_to_transform(curr_pose) 
                
                # 历史帧车体坐标 -> 全局 -> 当前参考帧车体坐标
                tm = reduce(
                    np.dot,
                    [ref_from_global, global_from_car],
                )

                curr_time = int(curr_obj['frame_name'].split("_")[-1])
                time_lag = ref_time - 1e-6 * curr_time

                sweep = {
                    "path": curr_lidar_path,
                    "transform_matrix": tm,
                    "time_lag": time_lag,
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        if split != 'test':
            # read boxes 
            TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
            annos = ref_obj['objects']
            num_points_in_gt = np.array([ann['num_points'] for ann in annos])
            gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
            
            if len(gt_boxes) != 0:
                # transform from Waymo to KITTI coordinate 
                # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
                # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
                # 内部约定 yaw = -pi/2 - heading（heading 顺时针，yaw 逆时针）
                gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
                # 交换长宽使内部顺序为 [x,y,z,dx=wid,dy=len,dz=hei,vel_x,vel_y,yaw]
                gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

            gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

            # filter boxes without lidar points 
            info['gt_boxes'] = gt_boxes[mask_not_zero, :].astype(np.float32)
            info['gt_names'] = gt_names[mask_not_zero].astype(str)

        infos.append(info)
    return infos

def sort_frame(frames):
    """按 (序列号, 帧号) 全局排序帧文件名列表。

    帧名形如 "seq_{n}_..._frame_{m}.pkl"，排序键为 n*1000+m，保证跨序列按序排列。

    Args:
        frames (list): 帧文件名列表。

    Returns:
        list: 排序后的帧文件名列表。
    """
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def get_available_frames(root, split):
    """列出某一 split 下所有实际存在的 lidar 帧，并按序返回。

    Args:
        root (str): 数据集根目录。
        split (str): 数据集切分。

    Returns:
        list: 排序后的帧文件名列表。
    """
    dir_path = os.path.join(root, split, 'lidar')
    available_frames = list(os.listdir(dir_path))

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(available_frames))
    return sorted_frames


def create_waymo_infos(root_path, split='train', nsweeps=1):
    """生成并落盘某个 split 的 infos pickle 文件。

    Args:
        root_path (str): 数据集根目录。
        split (str): 数据集切分（train/val/test）。
        nsweeps (int): 聚合的 sweep 数量。
    """
    frames = get_available_frames(root_path, split)

    waymo_infos = _fill_infos(
        root_path, frames, split, nsweeps
    )

    print(
        f"sample: {len(waymo_infos)}"
    )
    with open(
        os.path.join(root_path, "infos_"+split+"_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps)), "wb"
    ) as f:
        pickle.dump(waymo_infos, f)

def parse_args():
    """解析命令行参数（用于生成预测/真值结果二进制文件）。

    Returns:
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
    parser.add_argument("--path", type=str, default="data/Waymo/tfrecord_training")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--gt", action='store_true' )
    parser.add_argument("--tracking", action='store_true')
    args = parser.parse_args()
    return args


def reorganize_info(infos):
    """把 info 列表重排为 token -> info 的字典，便于按帧 token 快速查找。

    Args:
        infos (list): info 字典列表。

    Returns:
        dict: {token: info} 映射。
    """
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    with open(args.info_path, 'rb') as f:
        infos = pickle.load(f)
    
    if args.gt:
        _create_gt_detection(infos, tracking=args.tracking)
        exit() 

    infos = reorganize_info(infos)
    with open(args.path, 'rb') as f:
        preds = pickle.load(f)
    _create_pd_detection(preds, infos, args.result_path, tracking=args.tracking)
