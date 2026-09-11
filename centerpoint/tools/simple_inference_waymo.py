"""Waymo 逐帧离线推理脚本。

读取指定目录下的 Waymo 点云 pickle 文件，逐帧完成体素化与模型前向推理，
收集所有帧的 3D 检测框、类别与置信度，最终输出 detections.pkl（可选
visualization.pkl），并可用 open3d 在线可视化推理结果。

主要函数：
    - initialize_model: 根据配置构建检测器、加载权重并初始化体素生成器。
    - voxelization: 对点云做体素化（辅助函数）。
    - _process_inputs: 将体素化结果整理为模型输入字典。
    - run_model: 单帧前向推理并返回框、分数与类别。
    - process_example: 单帧推理的包装，校验输出字段一致性。

命令行参数：
    config           配置文件路径（位置参数）。
    --checkpoint     模型权重路径。
    --input_data_dir 存放逐帧点云 pickle 的输入目录（必填）。
    --output_dir     结果输出目录（必填）。
    --fp16           是否以 fp16 精度推理。
    --threshold      在线可视化时的显示置信度阈值。
    --visual         是否输出/展示可视化结果。
    --online         是否用 open3d 在线逐帧显示。
    --num_frame      最多推理帧数（-1 表示全部）。

输入约定：每个 pickle 文件包含以 'points' 为键的点云数组（Waymo 格式）。
"""
# modified from the single_inference.py by @muzi2045
from spconv.utils import VoxelGenerator as VoxelGenerator
from det3d.datasets.pipelines.loading import read_single_waymo
from det3d.datasets.pipelines.loading import get_obj
from det3d.torchie.trainer import load_checkpoint
from det3d.models import build_detector
from det3d.torchie import Config
from tqdm import tqdm 
import numpy as np
import pickle 
import open3d as o3d
import argparse
import torch
import time 
import os 

voxel_generator = None 
model = None 
device = None 

def initialize_model(args):
    """根据配置初始化模型与全局体素生成器。

    Args:
        args (argparse.Namespace): 命令行参数，含 config、checkpoint、fp16。

    Returns:
        torch.nn.Module: 已加载权重并切换到 eval 模式的模型。
    """
    global model, voxel_generator  
    cfg = Config.fromfile(args.config)
    # 以测试模式构建检测器（无训练配置）
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")
    # print(model)
    if args.fp16:
        print("cast model to fp16")
        model = model.half()

    model = model.cuda()
    model.eval()

    global device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    range = cfg.voxel_generator.range
    voxel_size = cfg.voxel_generator.voxel_size
    max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
    max_voxel_num = cfg.voxel_generator.max_voxel_num[1]
    # 按配置构建体素生成器（取最大体素数列表的第二个值作为上限）
    voxel_generator = VoxelGenerator(
        voxel_size=voxel_size,
        point_cloud_range=range,
        max_num_points=max_points_in_voxel,
        max_voxels=max_voxel_num
    )
    return model 

def voxelization(points, voxel_generator):
    """将点云体素化，返回体素特征、坐标与每体素点数。

    Args:
        points (np.ndarray): 输入点云。
        voxel_generator: 体素生成器对象。

    Returns:
        tuple: (体素特征, 体素坐标, 每体素点数)。
    """
    voxel_output = voxel_generator.generate(points)  
    voxels, coords, num_points = \
        voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']

    return voxels, coords, num_points  

def _process_inputs(points, fp16):
    """对单帧点云体素化并封装成模型输入字典。

    Args:
        points (np.ndarray): 输入点云。
        fp16 (bool): 是否将体素特征转为 fp16。

    Returns:
        dict: 包含 voxels/num_points/num_voxels/coordinates/shape 的输入字典。
    """
    voxels, coords, num_points = voxel_generator.generate(points)
    num_voxels = np.array([voxels.shape[0]], dtype=np.int32)
    grid_size = voxel_generator.grid_size
    # 坐标首列补 0 作为 batch 索引（模拟 batch=1）
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
    num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=device)

    if fp16:
        voxels = voxels.half()

    inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )

    return inputs 

def run_model(points, fp16=False):
    """对单帧点云执行前向推理。

    Args:
        points (np.ndarray): 输入点云。
        fp16 (bool): 是否以 fp16 精度推理。

    Returns:
        dict: 包含 boxes（3D 框）、scores（分数）、classes（类别）的字典。
    """
    with torch.no_grad():
        data_dict = _process_inputs(points, fp16)
        outputs = model(data_dict, return_loss=False)[0]

    return {'boxes': outputs['box3d_lidar'].cpu().numpy(),
        'scores': outputs['scores'].cpu().numpy(),
        'classes': outputs['label_preds'].cpu().numpy()}

def process_example(points, fp16=False):
    """单帧推理并校验输出结构的一致性。

    Args:
        points (np.ndarray): 输入点云。
        fp16 (bool): 是否以 fp16 精度推理。

    Returns:
        dict: run_model 的输出，键为 ('boxes', 'scores', 'classes')。
    """
    output = run_model(points, fp16)

    assert len(output) == 3
    assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
    num_objs = output['boxes'].shape[0]
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    return output    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--checkpoint", help="the path to checkpoint which the model read from", default=None, type=str
    )
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--visual', action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    print("Please prepare your point cloud in waymo format and save it as a pickle dict with points key into the {}".format(args.input_data_dir))
    print("One point cloud should be saved in one pickle file.")
    print("Download and save the pretrained model at {}".format(args.checkpoint))

    # 运行用户指定的初始化逻辑：构建模型并加载权重
    model = initialize_model(args)

    latencies = []
    visual_dicts = []
    pred_dicts = {}
    counter = 0 
    for frame_name in tqdm(sorted(os.listdir(args.input_data_dir))):
        # 达到指定的最大帧数后停止（-1 表示不限制）
        if counter == args.num_frame:
            break
        else:
            counter += 1 

        pc_name = os.path.join(args.input_data_dir, frame_name)
        points = pickle.load(open(pc_name, 'rb'))['points']
        # points = read_single_waymo(get_obj(pc_name))

        detections = process_example(points, args.fp16)

        if args.visual and args.online:
            # 在线模式：用 open3d 逐帧展示点云与 3D 框
            pcd = o3d.geometry.PointCloud()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            visual += plot_boxes(detections, args.threshold)

            o3d.visualization.draw_geometries(visual)
        elif args.visual:
            # 离线模式：收集点云与检测结果供后续可视化
            visual_dicts.append({'points': points, 'detections': detections})

        pred_dicts.update({frame_name: detections})

    if args.visual:
        with open(os.path.join(args.output_dir, 'visualization.pkl'), 'wb') as f:
            pickle.dump(visual_dicts, f)

    with open(os.path.join(args.output_dir, 'detections.pkl'), 'wb') as f:
        pickle.dump(pred_dicts, f)
