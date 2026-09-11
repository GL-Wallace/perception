"""nuScenes 跟踪评估的主入口脚本。

读取第一阶段检测结果（checkpoint 中的 JSON），按帧组织为时间序列，逐帧调用
PubTracker.step_centertrack 得到跟踪结果，并以 nuScenes 官方格式写出
tracking_result.json，最后调用 TrackingEval 计算跟踪指标（AMOTA/AMOTP 等）。

主要函数：
    - save_first_frame: 生成帧级元数据（首帧标记与时间戳）。
    - main: 逐帧跟踪并写出结果。
    - eval_tracking / eval: 调用官方评测计算指标。
    - test_time: 多次运行统计最快 FPS。

与其他模块关系：
    - 依赖 pub_tracker.PubTracker 做跟踪；
    - 依赖 nuscenes SDK 读取数据与官方评测。
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
from pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits

def parse_args():
    """解析命令行参数（工作目录、checkpoint、是否匈牙利匹配、max_age 等）。

    Returns:
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="data/nuScenes")
    parser.add_argument("--version", type=str, default='v1.0-trainval')
    parser.add_argument("--max_age", type=int, default=3)

    args = parser.parse_args()

    return args


def save_first_frame():
    """生成 frames_meta.json：记录每帧的 token、时间戳与是否为场景首帧。"""
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # sample['prev'] 为空表示它是该场景序列的第一帧。
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def main():
    """逐帧执行跟踪，输出 nuScenes 官方格式的 tracking_result.json。

    Returns:
        float: 平均处理速度（FPS）。
    """
    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)

    with open(args.checkpoint, 'rb') as f:
        predictions=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in range(size):
        token = frames[i]['token']

        # 每个新视频序列开始时重置跟踪器，避免跨场景错误关联。
        if frames[i]['first']:
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        # 距上一帧的时间间隔（单位：秒，save_first_frame 中已把微秒换算为秒），供速度平移使用。
        time_lag = (frames[i]['timestamp'] - last_time_stamp) 
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token]

        outputs = tracker.step_centertrack(preds, time_lag)
        annos = []

        # 只输出 active 的轨迹；丢失但仍在保留期内的轨迹不输出到当前帧。
        for item in outputs:
            if item['active'] == 0:
                continue 
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    
    end = time.time()

    second = (end-start) 

    speed=size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking():
    """对刚生成的 tracking_result.json 调用官方评测。"""
    args = parse_args()
    eval(os.path.join(args.work_dir, 'tracking_result.json'),
        "val",
        args.work_dir,
        args.root
    )

def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    """用 nuScenes 官方 TrackingEval 计算 AMOTA/AMOTP 等跟踪指标。

    Args:
        res_path (str): 跟踪结果 JSON 路径。
        eval_set (str): 评测集（val/test）。
        output_dir (str): 结果输出目录。
        root_path (str): nuScenes 数据集根目录。
    """
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs

    
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time():
    """多次运行 main 统计最快的 FPS（用于性能测试）。"""
    speeds = []
    for i in range(3):
        speeds.append(main())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    save_first_frame()
    main()
    # test_time()
    eval_tracking()
