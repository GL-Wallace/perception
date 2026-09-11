"""CenterPoint 数据预处理流水线（PIPELINES 注册的算子）。

负责点云/标注的增强（Preprocess）、体素化（Voxelization）与训练目标构造（AssignLabel）。
AssignLabel 对应论文 Sec 3.2：把 GT 3D 中心投影到下采样后的特征图网格，
生成中心热图(hm)、回归目标(anno_box)、中心索引(ind)、有效掩码(mask)与类别(cat)。

主要类：
    - Preprocess: 训练时做地面真值筛选、GT 采样与随机翻转/旋转/缩放/平移等增强。
    - Voxelization: 把点云转为体素/柱体表示，供 backbone 消费。
    - AssignLabel: 构造 CenterPoint 的热图与回归监督（论文 Sec 3.2）。

主要函数：
    - _dict_select: 递归地用索引筛选字典内数组。
    - drop_arrays_by_name: 返回需要剔除的 GT 名称索引。
    - flatten: 沿第 0 维拼接数组列表。
    - merge_multi_group_label: 合并多 task 的类别标签并加上类别偏移。

与其他模块的关系：
    - 各算子通过 PIPELINES 注册表注册，由 det3d/datasets 的 dataset 在加载样本时依次调用 __call__。
    - 依赖 det3d.core.utils.center_utils 的 gaussian_radius/draw_umich_gaussian 与
      det3d.core.sampler.preprocess 的几何增强函数。
"""

import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES


def _dict_select(dict_, inds):
    """递归地按索引 inds 筛选字典中的数组值（支持嵌套字典）。

    Args:
        dict_ (dict): 待筛选的字典，值为 ndarray 或嵌套 dict。
        inds (ndarray): 保留的索引。
    """
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    """返回需要剔除的 GT 下标：名称属于 used_classes（如 DontCare/ignore）的目标。

    Args:
        gt_names (list/ndarray): GT 目标的名称列表。
        used_classes (list): 需要剔除的类别名集合。

    Returns:
        ndarray: 待剔除目标的索引。
    """
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    """点云/标注的预处理与训练增强算子。

    训练时负责：筛选非法 GT、过滤点数不足的 GT、可选 GT 数据库采样（copy-paste），
    并对点云与标注做随机翻转、全局旋转、缩放、平移等增强；测试时仅提取点云。
    """

    def __init__(self, cfg=None, **kwargs):
        """读取增强相关配置。

        Args:
            cfg: 配置对象，含 shuffle_points/mode/class_names 及各增强噪声参数。
        """
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):
        """执行预处理/增强。

        Args:
            res (dict): 单样本数据字典，含 lidar 点云与标注。
            info (dict): 样本元信息。

        Returns:
            Tuple[dict, dict]: 增强后的 (res, info)。
        """

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            # 剔除 DontCare/ignore/UNKNOWN 等不参与监督的 GT。
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            # 过滤框内点云数过少的 GT，避免监督退化。
            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= min_points_in_gt
                _dict_select(gt_dict, mask)

            # 仅保留属于训练类别集合的 GT。
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                # GT 数据库采样（copy-paste），把采样框与其点云拼入当前帧以缓解类别不均衡。
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )


                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            # 类别映射为 class_names 中的下标 + 1（0 留给背景）。
            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            # 依次做随机翻转、全局旋转、缩放、平移，保证点云与框同步变换。
            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            )
        elif self.no_augmentation:
            # 关闭增强时仅保留训练类别集合内的 GT。
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info


@PIPELINES.register_module
class Voxelization(object):
    """把点云转成体素/柱体(pillar)表示，供 backbone 消费。

    训练与测试可使用不同的最大体素数量；可选为测试生成 y 翻转、x 翻转、
    双翻转三种点云的体素，用于 double flip 测试时增强。
    """

    def __init__(self, **kwargs):
        """读取体素化配置并构造 VoxelGenerator。

        Args:
            kwargs: 含 cfg 配置对象（range/voxel_size/max_points_in_voxel/max_voxel_num 等）。
        """
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        # 训练/测试可用不同体素上限：若配置为 int 则两者相同。
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num

        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        """执行体素化，把体素写入 res["lidar"]["voxels"]。

        Args:
            res (dict): 单样本数据字典。
            info (dict): 样本元信息。

        Returns:
            Tuple[dict, dict]: 处理后的 (res, info)。
        """
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            gt_dict = res["lidar"]["annotations"]
            # 训练时再过滤一次中心超出 BEV 范围的 GT（仅用 x/y 维度）。
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        # 记录体素数据与网格 shape/range/size，供下游 AssignLabel 计算 feature map 尺寸。
        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        # 仅为测试生成三种翻转点云的体素，推理时在 CenterHead.predict 中做融合。
        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info

def flatten(box):
    """沿第 0 维拼接数组列表。"""
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    """合并多 task 的类别标签，为每个 task 加上前序 task 的累计类别偏移。

    Args:
        gt_classes (List[ndarray]): 每个 task 的类别下标列表。
        num_classes_by_task (List[int]): 每个 task 的类别数。

    Returns:
        ndarray: 偏移并拼接后的全局类别标签。
    """
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    """构造 CenterPoint 的训练监督目标（论文 Sec 3.2）。

    把每个 GT 的中心投影到下采样后的特征图网格，生成：
        - hm: 每个 task、每个类别一张中心热图（中心处放置 2D 高斯，论文 Sec 3.1）。
        - anno_box: 回归目标，顺序为 offset(2) + z(1) + log(dim)(3) + vel(2) + sin/cos(yaw)(2)。
        - ind: GT 中心的展平索引（ind = y * W + x）。
        - mask: 有效目标掩码（1 表示该槽位存在 GT）。
        - cat: 每个 GT 的类别 id。
    另生成 gt_boxes_and_cls 供两阶段精炼使用。
    """

    def __init__(self, **kwargs):
        """读取目标分配配置。

        Args:
            kwargs: 含 cfg 配置对象（out_size_factor/target_assigner/gaussian_overlap/
                max_objs/min_radius 等）。
        """
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.cfg = assigner_cfg

    def __call__(self, res, info):
        """为训练样本生成热图与回归监督，写入 res["lidar"]["targets"]。

        Args:
            res (dict): 单样本数据字典。
            info (dict): 样本元信息。

        Returns:
            Tuple[dict, dict]: 处理后的 (res, info)。
        """
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        if res["mode"] == "train":
            # 计算 backbone 输出 feature map 的尺寸（体素网格除以下采样倍率）。
            if 'voxels' in res['lidar']:
                grid_size = res["lidar"]["voxels"]["shape"] 
                pc_range = res["lidar"]["voxels"]["range"]
                voxel_size = res["lidar"]["voxels"]["size"]
                feature_map_size = grid_size[:2] // self.out_size_factor
            else:
                pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                grid_size = np.round(grid_size).astype(np.int64)

            feature_map_size = grid_size[:2] // self.out_size_factor

            gt_dict = res["lidar"]["annotations"]

            # 按 task 重新组织 GT：找出每个类别的 GT 下标（类别 id 含跨 task 的全局偏移）。
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    # 类别 id 去掉前序 task 的累计偏移，得到 task 内 0 起始的类别 id。
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # 把旋转角限制到 [-pi, pi]，避免角度环绕问题。
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                # 热图 shape 为 (类别数, feature_map 高, feature_map 宽)，注意后面索引用的是 (cls, y, x)。
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # 回归目标共 10 维：[reg(x,y), z, log(dim)(w,l,h), vx, vy, sin(yaw), cos(yaw)]。
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    # task 内 0 起始的类别 id（gt_classes 存的是 1 起始，故减 1）。
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    # 把物理尺寸换算到 feature 格：除以 voxel_size 与下采样倍率。
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        # 论文 Sec 3.1：按物体 BEV 尺寸与 overlap 计算高斯半径，并取最小半径下限。
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # 注意坐标约定：gt_boxes 前三维为 (x, y, z)，中心投影到 BEV 网格。
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # 丢弃落在 feature map 之外的目标，避免绘制热图时越界。
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        # 展平索引：ind = y * W + x，供热图/回归损失在展开后的特征上 gather。
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            # 中心 offset 用连续坐标 ct 与取整坐标 (x, y) 之差，回归亚像素精度。
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # 供两阶段精炼代码使用：拼出全局框与类别标签。
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # 统一字段顺序为 x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name。
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info

