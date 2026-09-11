"""nuScenes 推理结果可视化 demo。

加载固定配置与固定 checkpoint，对验证集逐帧做前向推理，将点云（鸟瞰视角）与
真值框、预测框绘制成图片，再用 OpenCV 将全部图片合成为视频。

主要函数：
    - convert_box: 将 info 中的 gt_boxes/gt_names 转换为检测结果字典格式。
    - main: 推理与可视化主流程，输出 demo/*.png 图片与 video.avi 视频。

注意：
    本脚本演示用途较强，配置文件路径与 checkpoint 路径硬编码在代码中，
    推理结果通过 tools.demo_utils.visual 渲染为 BEV 图。
"""
import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import load_checkpoint
import pickle 
import time 
from matplotlib import pyplot as plt 
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader
import matplotlib.cm as cm
import subprocess
import cv2
from tools.demo_utils import visual 
from collections import defaultdict

def convert_box(info):
    """将 info 中的真值框转换为 detection 结构，便于复用统一的可视化渲染。

    Args:
        info (dict): 单帧 info，需包含 'gt_boxes' 与 'gt_names' 字段。

    Returns:
        dict: 包含 box3d_lidar、label_preds、scores 的字典；其中 label_preds
            与 scores 为占位值（均为 0 / 1），仅用于走通检测结果的渲染流程。
    """
    boxes =  info["gt_boxes"].astype(np.float32)
    names = info["gt_names"]

    assert len(boxes) == len(names)

    detection = {}

    detection['box3d_lidar'] = boxes

    # 占位值：真值框没有类别标签与置信度，填 0/1 以便复用检测可视化接口
    detection['label_preds'] = np.zeros(len(boxes)) 
    detection['scores'] = np.ones(len(boxes))

    return detection 

def main():
    """推理与可视化主流程。

    加载固定配置与 checkpoint，遍历验证集做推理，将结果逐帧渲染为 BEV 图并
    合成为视频：
    1. 构建模型、数据集与数据加载器（batch=1）；
    2. 加载权重并切到 eval 模式；
    3. 逐帧推理，收集点云、真值与检测结果；
    4. 调用 tools.demo_utils.visual 渲染图片；
    5. 用 OpenCV 将图片合成为 video.avi。
    """
    cfg = Config.fromfile('configs/nusc/pp/nusc_centerpoint_pp_02voxel_two_pfn_10sweep_demo.py')
    
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    dataset = build_dataset(cfg.data.val)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_kitti,
        pin_memory=False,
    )

    # 加载 checkpoint（先在 CPU 上加载，后续迁到 GPU）
    checkpoint = load_checkpoint(model, 'work_dirs/centerpoint_pillar_512_demo/latest.pth', map_location="cpu")
    model.eval()

    model = model.cuda()

    cpu_device = torch.device("cpu")

    points_list = [] 
    gt_annos = [] 
    detections  = [] 

    for i, data_batch in enumerate(data_loader):
        info = dataset._nusc_infos[i]
        gt_annos.append(convert_box(info))

        # 取点云前 xyz 三列（去掉首列的反射强度等通道）
        points = data_batch['points'][:, 1:4].cpu().numpy()
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=0,
            )
        for output in outputs:
            # 除 metadata 外的张量统一搬到 CPU
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.append(output)

        # 转置成 (3, N) 形式，便于 BEV 渲染
        points_list.append(points.T)
    
    print('Done model inference. Please wait a minute, the matplotlib is a little slow...')
    
    for i in range(len(points_list)):
        visual(points_list[i], gt_annos[i], detections[i], i)
        print("Rendered Image {}".format(i))
    
    image_folder = 'demo'
    video_name = 'video.avi'

    # 按帧序号排序取回所有渲染图片
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda img_name: int(img_name.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 以首帧尺寸创建视频写入器，帧率为 1
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    cv2_images = [] 

    for image in images:
        cv2_images.append(cv2.imread(os.path.join(image_folder, image)))

    for img in cv2_images:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    print("Successfully save video in the main folder")

if __name__ == "__main__":
    main()
