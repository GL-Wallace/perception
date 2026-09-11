"""Waymo 跟踪超参数网格搜索脚本。

对各类别（VEHICLE/PEDESTRIAN/CYCLIST）分别遍历 score（新轨迹分数阈值）与
dist（最近点匹配距离阈值）的不同取值，为每个组合调用 tools/waymo_tracking/test.py
跑一次跟踪并保存结果，用于挑选最佳阈值（对应 tracker.py 的 score_thresh 与
max_dist 参数）。

主要逻辑：
    - scores/dists 定义各类别的候选网格；
    - 双层循环组合 score 与 dist 并拼装命令行。
"""

import os
import numpy as np 

# 各类别新轨迹分数阈值的候选网格。
scores = {
    0: np.arange(0.4, 0.8, 0.02),
    1: np.arange(0.4, 0.8, 0.02),
    2: np.arange(0.4, 0.8, 0.02)
}

# 各类别最近点匹配距离阈值的候选网格。
dists = {
    0: np.arange(0.4, 0.8, 0.04),
    1: np.arange(0.1, 0.5, 0.04),
    2: np.arange(0.3, 0.7, 0.04)
}

for label in range(3):
    score_list = scores[label]
    dist_list = dists[label]

    # 遍历当前类别下 score 与 dist 的所有组合。
    for score in score_list:
        for dist in dist_list:
            # 用类别/score/dist 命名工作目录，分开保存各组结果。
            work_dir = "waymo_track/label_{}_score_{}_max_age_{}_dist_{}".format(label, score, dist)

            # 拼装 test.py 命令行，并把日志重定向到 stats.txt。
            cmd=("python tools/waymo_tracking/test.py " + 
                "--checkpoint /home/tianweiy/base/work_dirs/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo/prediction.pkl"
                "--work_dir {}".format(work_dir) + 
                "  --info_path data/Waymo/infos_val_02sweeps_filter_zero_gt.pkl" + 
                "--vehicle {}  --pedestrian {}  --cyclist".format(dist, dist, dist) +
                "--score_thresh {}".format(score) + 
                "--name {}".format(label) + 
                "> {}/stats.txt ".format(work_dir)
            )[0]

            print(cmd)
            os.system(cmd)
