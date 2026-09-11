"""单帧点云 ROS 推理节点。

订阅 lidar 点云话题，将每帧点云单独体素化并送入 CenterPoint 推理，最后以
BoundingBoxArray 形式发布检测结果。与 multi_sweep_inference_ros.py 的区别在于
本脚本不做多帧聚合，逐帧独立推理。

主要函数：
    - yaw2quaternion: yaw 角转四元数。
    - get_annotations_indices / remove_low_score_nu: 按类别与分数过滤检测结果。
    - Processor_ROS: 封装模型加载、体素化与单帧推理。
    - rslidar_callback: lidar 话题回调。

订阅的 lidar 话题（默认取第 6 个候选）：/velodyne_points、
/top/rslidar_points、/points_raw、/lidar_protector/merged_cloud、/merged_cloud、
/lidar_top、/roi_pclouds；检测结果发布话题为 pp_boxes（BoundingBoxArray）。
"""

import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import torch
import time 

from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from pyquaternion import Quaternion

from det3d import __version__, torchie
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.core.input.voxel_generator import VoxelGenerator

def yaw2quaternion(yaw: float) -> Quaternion:
    """yaw 角转绕 z 轴旋转的四元数。

    Args:
        yaw (float): 绕 z 轴的旋转角（弧度）。

    Returns:
        Quaternion: 对应的四元数。
    """
    return Quaternion(axis=[0,0,1], radians=yaw)

def get_annotations_indices(types, thresh, label_preds, scores):
    """筛选出类别等于 types 且分数高于 thresh 的样本下标。

    Args:
        types (int): 目标类别序号。
        thresh (float): 分数阈值。
        label_preds (np.ndarray): 预测类别数组。
        scores (np.ndarray): 预测分数数组。

    Returns:
        list: 满足条件的样本下标列表。
    """
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    """按 nuScenes 各类别的分数阈值过滤检测结果。

    Args:
        image_anno (dict): 模型输出字典，张量字段可 detach。
        thresh (float): 未直接使用的阈值（各类阈值在函数内硬编码）。

    Returns:
        dict: 过滤并拼接后的标注字典（metadata 字段被跳过）。
    """
    img_filtered_annotations = {}
    label_preds_ = image_anno["label_preds"].detach().cpu().numpy()
    scores_ = image_anno["scores"].detach().cpu().numpy()
    
    # 各 nuScenes 类别使用不同的分数阈值
    car_indices =                  get_annotations_indices(0, 0.4, label_preds_, scores_)
    truck_indices =                get_annotations_indices(1, 0.4, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(2, 0.4, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(3, 0.3, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(4, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(6, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(7, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(8, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(9, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    """ROS 单帧推理处理器。

    维护体素生成器与模型，对每帧输入点云做体素化并推理。关键属性：
        - net: 已加载权重并切换到 eval 模式的模型；
        - voxel_generator: 体素生成器；
        - inputs: 最近一次推理的输入字典。
    """
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None
        self.voxel_generator = None
        self.inputs = None
        
    def initialize(self):
        """初始化：读取配置并构建模型与体素生成器。"""
        self.read_config()
        
    def read_config(self):
        """读取配置、构建模型并初始化体素生成器。"""
        config_path = self.config_path
        cfg = Config.fromfile(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        self.net.load_state_dict(torch.load(self.model_path)["state_dict"])
        self.net = self.net.to(self.device).eval()

        self.range = cfg.voxel_generator.range
        self.voxel_size = cfg.voxel_generator.voxel_size
        self.max_points_in_voxel = cfg.voxel_generator.max_points_in_voxel
        self.max_voxel_num = cfg.voxel_generator.max_voxel_num
        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

    def run(self, points):
        """对单帧点云体素化并前向推理。

        Args:
            points (np.ndarray): 输入点云（单帧）。

        Returns:
            tuple: (scores, boxes_lidar, types)，即分数、lidar 坐标下的 3D 框
                （最后一列朝向已做 -yaw - pi/2 变换）与类别。
        """
        t_t = time.time()
        print(f"input points shape: {points.shape}")
        num_features = 5        
        self.points = points.reshape([-1, num_features])
        self.points[:, 4] = 0 # 时间戳通道置 0
        
        voxels, coords, num_points = self.voxel_generator.generate(self.points)
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
        grid_size = self.voxel_generator.grid_size
        # 坐标首列补 0 作为 batch 索引
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values = 0)
        
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        num_voxels = torch.tensor(num_voxels, dtype=torch.int32, device=self.device)
        
        self.inputs = dict(
            voxels = voxels,
            num_points = num_points,
            num_voxels = num_voxels,
            coordinates = coords,
            shape = [grid_size]
        )
        torch.cuda.synchronize()
        t = time.time()

        with torch.no_grad():
            outputs = self.net(self.inputs, return_loss=False)[0]
    
        # print(f"output: {outputs}")
        
        torch.cuda.synchronize()
        print("  network predict time cost:", time.time() - t)

        outputs = remove_low_score_nu(outputs, 0.45)

        boxes_lidar = outputs["box3d_lidar"].detach().cpu().numpy()
        print("  predict boxes:", boxes_lidar.shape)

        scores = outputs["scores"].detach().cpu().numpy()
        types = outputs["label_preds"].detach().cpu().numpy()

        # 朝向约定切换：yaw 取反并旋转 -pi/2 对齐输出坐标系
        boxes_lidar[:, -1] = -boxes_lidar[:, -1] - np.pi / 2

        print(f"  total cost time: {time.time() - t_t}")

        return scores, boxes_lidar, types

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    """从结构化点云数组提取 x/y/z 为 (N, 5) 的点数组。

    Args:
        cloud_array: 结构化点云数组（含 x/y/z 字段）。
        remove_nans (bool): 是否剔除含 NaN/Inf 坐标的点。
        dtype: 输出数组数据类型。

    Returns:
        np.ndarray: 形状 (N, 5) 的点数组，前三列分别为 x/y/z。
    """
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    """将点数组封装为 sensor_msgs.PointCloud2 消息。

    Args:
        points_sum (np.ndarray): 形状 (N, 3) 的点坐标数组。
        stamp: 消息时间戳。
        frame_id: 坐标系名称。

    Returns:
        PointCloud2: 仅含 x/y/z 三个字段的点云消息。
    """
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    """lidar 话题回调：收帧、触发推理并发布检测框。"""
    t_t = time.time()
    arr_bbox = BoundingBoxArray()

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    scores, dt_box_lidar, types = proc_1.run(np_p)

    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            q = yaw2quaternion(float(dt_box_lidar[i][8]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]           
            bbox.pose.position.x = float(dt_box_lidar[i][0])
            bbox.pose.position.y = float(dt_box_lidar[i][1])
            bbox.pose.position.z = float(dt_box_lidar[i][2])
            bbox.dimensions.x = float(dt_box_lidar[i][4])
            bbox.dimensions.y = float(dt_box_lidar[i][3])
            bbox.dimensions.z = float(dt_box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types[i])
            arr_bbox.boxes.append(bbox)
    print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = msg.header.frame_id
    arr_bbox.header.stamp = msg.header.stamp
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)
   
if __name__ == "__main__":

    global proc
    ## CenterPoint
    config_path = 'configs/centerpoint/nusc_centerpoint_pp_02voxel_circle_nms_demo.py'
    model_path = 'models/last.pth'

    proc_1 = Processor_ROS(config_path, model_path)
    
    proc_1.initialize()
    
    rospy.init_node('centerpoint_ros_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/top/rslidar_points",
                        "/points_raw", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds"]
    
    sub_ = rospy.Subscriber(sub_lidar_topic[5], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)
    
    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)

    print("[+] CenterPoint ros_node has started!")    
    rospy.spin()