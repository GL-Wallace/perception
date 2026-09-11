"""多模型预测结果融合（ensemble + NMS）脚本。

读取指定目录下多个模型的 nuScenes 预测 pickle 文件，将同一 sample_token 的
预测框合并，按类别在全局坐标下做旋转 NMS 去重后，输出 result.json 并调用
nuScenes 官方评估器计算检测指标。

主要函数：
    - parse_args: 解析命令行参数。
    - get_sample_data: 将预测项转换为 Box 列表与分数。
    - reorganize_boxes: 将 nuScenes Box 转换为 lidar 坐标下的 center/size/yaw。
    - reorganize_pred_by_class: 按 detection_name 归组预测。
    - concatenate_list: 展平列表。
    - filter_pred_by_class: 按大类（small/large）过滤预测（本脚本未调用）。
    - get_pred: 读取单个预测 pickle。
    - main: 融合 + NMS + 评估主流程。

命令行参数：
    ensemble_dir     存放所有预测 pickle 的目录（位置参数）。
    --output_path    融合结果的输出路径。
    --data_root      nuScenes 数据根目录。

注意：
    ENS_CLASS/SMALL_CLASS/LARGE_CLASS 为按目标尺寸划分的类别常量。
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pickle 
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box, RadarPointCloud
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.geometry_utils import points_in_box
from functools import reduce
from tqdm import tqdm
from det3d.core import box_torch_ops
from collections import defaultdict 
import torch 
import glob 

def parse_args():
    """解析融合脚本的命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象，含 ensemble_dir、output_path、
            data_root（默认 'data/nuScenes/v1.0-trainval'）。
    """
    parser = argparse.ArgumentParser(description="Ensemble Models")
    parser.add_argument("ensemble_dir", help="path to a dir that contains all prediction file")
    parser.add_argument("--output_path", help="the path to save ensemble output")    
    parser.add_argument("--data_root", type=str, default="data/nuScenes/v1.0-trainval") 
    
    args = parser.parse_args()

    return args


def get_sample_data(pred):
    """将同一类别的预测项转换为 Box 列表与分数数组。

    Args:
        pred (list): 预测项列表，每项含 translation/size/rotation/detection_name/
            detection_score。

    Returns:
        tuple: (top_boxes, top_scores)，即中心/尺寸/朝向组成的 numpy 数组与分数。
    """
    box_list = [] 
    score_list = [] 
    pred = pred.copy() 

    for item in pred:    
        box =  Box(item['translation'], item['size'], Quaternion(item['rotation']),
                name=item['detection_name'])
        score_list.append(item['detection_score'])
        box_list.append(box)

    top_boxes = reorganize_boxes(box_list)
    top_scores = np.array(score_list).reshape(-1)

    return top_boxes, top_scores

def reorganize_boxes(box_lidar_nusc):
    """将 nuScenes Box 列表转换为 lidar 坐标系下的 center/size/yaw 数组。

    Args:
        box_lidar_nusc (list): 长度为 N 的 Box 列表。

    Returns:
        np.ndarray: 形状 (N, 7) 的数组，列为 (x, y, z, w, l, h, yaw)。
    """
    rots = []
    centers = []
    wlhs = []
    for i, box_lidar in enumerate(box_lidar_nusc):
        # 取旋转矩阵的第一列（前向轴），由 atan2 反解出 yaw
        v = np.dot(box_lidar.rotation_matrix, np.array([1, 0, 0]))
        rot = np.arctan2(v[1], v[0])

        rots.append(-rot- np.pi / 2)
        centers.append(box_lidar.center)
        wlhs.append(box_lidar.wlh)

    rots = np.asarray(rots)
    centers = np.asarray(centers)
    wlhs = np.asarray(wlhs)
    gt_boxes_lidar = np.concatenate([centers.reshape(-1,3), wlhs.reshape(-1,3), rots[..., np.newaxis].reshape(-1,1) ], axis=1)
    
    return gt_boxes_lidar

def reorganize_pred_by_class(pred):
    """按 detection_name 将预测项归组。

    Args:
        pred (list): 预测项列表。

    Returns:
        defaultdict(list): 类别名到预测项列表的映射。
    """
    ret_dicts = defaultdict(list)
    for item in pred: 
        ret_dicts[item['detection_name']].append(item) 

    return ret_dicts

def concatenate_list(lists):
    """将多个列表展平拼接为一个列表。

    Args:
        lists (list): 由多个列表组成的列表。

    Returns:
        list: 拼接后的列表。
    """
    ret = []
    for l in lists:
        ret += l 

    return ret 

ENS_CLASS = ['car', 'truck', 'bus', 'construction_vehicle', 'bicycle']
SMALL_CLASS = ['pedestrian', 'barrier', 'traffic_cone', 'motorcycle']
LARGE_CLASS = ['trailer']
ALL_CLASS = ['car', 'truck', 'bus', 'construction_vehicle', 'bicycle', 'pedestrian', 'barrier', 'traffic_cone', 'motorcycle', 'trailer']

def filter_pred_by_class(preds, small=False, large=False):
    """按大类过滤预测结果（按 small/large 保留子集）。

    Args:
        preds (dict): token 到预测项列表的映射。
        small (bool): 为 True 时过滤掉 LARGE_CLASS。
        large (bool): 为 True 时过滤掉 SMALL_CLASS。

    Returns:
        dict: 过滤后的 token 到预测项列表的映射。
    """
    ret_dict = {} 
    for token, pred in preds.items():
        filtered = []

        for item in pred:
            assert item['detection_name'] in ALL_CLASS

            if small:
                if item['detection_name'] not in LARGE_CLASS:
                    filtered.append(item)
            elif large:
                if item['detection_name'] not in SMALL_CLASS:
                    filtered.append(item)

        ret_dict[token] = filtered

    return ret_dict 

def get_pred(path):
    """读取单个预测 pickle 文件。

    Args:
        path (str): pickle 文件路径。

    Returns:
        dict: token 到预测项列表的预测字典。
    """
    with open(path, 'rb') as f:
        pred=pickle.load(f)

    return pred

def main():
    """融合多模型预测并做 NMS 去重，最后调用 nuScenes 评估器评估。

    步骤：
    1. 收集 ensemble_dir 下所有 pickle 预测；
    2. 按 sample_token 合并多个模型的预测项；
    3. 按类别做旋转 NMS 去重；
    4. 输出 result.json 并用 NuScenesEval 计算指标。
    """
    args = parse_args()

    pred_paths = glob.glob(os.path.join(args.ensemble_dir, '*.pkl'))
    print(pred_paths)

    preds = []
    for path in pred_paths:
        preds.append(get_pred(path))

    merged_predictions = {}
    for token in preds[0].keys():
        # 收集所有模型在该 token 下的预测
        annos = [pred[token] for pred in preds]

        merged_predictions[token] = concatenate_list(annos) 

    predictions = merged_predictions
    
    print("Finish Merging")

    nusc_annos = {
        "results": {},
        "meta": None,
    }

    for sample_token, prediction in tqdm(predictions.items()):
        annos = []

        # 按类别归组预测
        pred_dicts = reorganize_pred_by_class(prediction)

        for name, pred in pred_dicts.items():
            # 转换到全局坐标（此处即 lidar 坐标下的 center/size/yaw）
            top_boxes, top_scores = get_sample_data(pred)

            with torch.no_grad():
                top_boxes_tensor = torch.from_numpy(top_boxes)
                # 调整为 NMS 需要的字段顺序 (x, y, z, w, l, h, yaw)
                boxes_for_nms = top_boxes_tensor[:, [0, 1, 2, 4, 3, 5, -1]]
                boxes_for_nms[:, -1] = boxes_for_nms[:, -1] + np.pi /2 
                top_scores_tensor = torch.from_numpy(top_scores)

                selected = box_torch_ops.rotate_nms(boxes_for_nms, top_scores_tensor, 
                    pre_max_size=None,
                    post_max_size=50,  
                    iou_threshold=0.2,
                ).numpy()
        
            pred = [pred[s] for s in selected]

            annos.extend(pred)

        nusc_annos["results"].update({sample_token: annos})

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": True,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'result.json'), "w") as f:
        json.dump(nusc_annos, f)
    
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=True)
    cfg = config_factory("cvpr_2019")
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=os.path.join(args.work_dir, 'result.json'),
        eval_set='val',
        output_dir=args.work_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=0,)


if __name__ == "__main__":
    main()
