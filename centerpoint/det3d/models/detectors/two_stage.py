"""两阶段检测器：在单阶段检测基础之上追加第一阶段输出的精炼。

TwoStageDetector 组合一个单阶段检测器（single_det，如 VoxelNet/PointPillars）、
若干第二阶段特征提取模块（second_stage，如 BEV 特征提取器）以及一个 roi_head
做最终分类与回归。对应论文 Sec 3.4 的两阶段 refinement：从一阶段估计框的面
中心采样点特征，经 MLP 预测 IoU 置信度并对 box 做精炼。

主要类：
    - TwoStageDetector: 两阶段检测器。
"""

from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 

@DETECTORS.register_module
class TwoStageDetector(BaseDetector):
    """两阶段检测器，在一阶段预测基础上追加第二阶段的 roi_head 精炼。"""

    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        roi_head, 
        NMS_POST_MAXSIZE,
        num_point=1,
        freeze=False,
        use_final_feature=False,
        **kwargs
    ):
        """构造两阶段检测器，实例化单阶段检测器与第二阶段模块。

        Args:
            first_stage_cfg (dict): 第一阶段单阶段检测器配置。
            second_stage_modules (list): 第二阶段特征提取模块配置列表。
            roi_head (dict): ROI 头（分类 / 回归）配置。
            NMS_POST_MAXSIZE (int): 每帧保留的 ROI 最大数量，用于对齐到固定矩阵。
            num_point (int): 每个 box 采样的点数（1 表示仅中心，5 表示中心 + 四边中点）。
            freeze (bool): 是否冻结第一阶段网络。
            use_final_feature (bool): 是否使用检测头最终特征作为 BEV 特征。
            **kwargs: 传递给第一阶段检测器的额外参数。
        """
        super(TwoStageDetector, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)
        self.NMS_POST_MAXSIZE = NMS_POST_MAXSIZE

        if freeze:
            print("Freeze First Stage Network")
            # 分两步训练：先冻结第一阶段权重，仅训练第二阶段。 
            self.single_det = self.single_det.freeze()
        self.bbox_head = self.single_det.bbox_head

        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            self.second_stage.append(builder.build_second_stage_module(module))

        self.roi_head = builder.build_roi_head(roi_head)

        self.num_point = num_point
        self.use_final_feature = use_final_feature

    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        """将第二阶段 ROI 损失并入一阶段损失字典。

        Args:
            one_stage_loss (dict): 一阶段损失字典，含 'loss' 与 'roi_*_loss' 列表。
            roi_loss: ROI 头的总损失标量。
            tb_dict (dict): 含 rcnn_loss_reg / rcnn_loss_cls 的日志字典。

        Returns:
            合并后的损失字典。
        """
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    def get_box_center(self, boxes):
        """根据 box 计算用于第二阶段采样的中心点集合。

        当 num_point==1 时使用 box 中心；num_point==5 时额外加入 BEV 框四条边的
        中点（论文 Sec 3.4 对框面中心采样）。

        Args:
            boxes (list): 一阶段预测结果列表，元素含 'box3d_lidar'。

        Returns:
            list: 每个样本的中心点张量。
        """
        # boxes 为样本列表。
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])
                
            elif self.num_point == 5:
                # 论文 Sec 3.4：对 box 中心与 BEV 框四条边中点共 5 个位置采样。
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                # 由 BEV 中心、尺寸与朝向角计算 2D 框四角。
                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                # BEV 框四条边的中点（前 / 后 / 左 / 右）。
                front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1) 

                points = torch.cat([box['box3d_lidar'][:, :3], front_middle, back_middle, left_middle, \
                    right_middle], dim=0)

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def reorder_first_stage_pred_and_feature(self, first_pred, example, features):
        """将一阶段预测与采样特征按固定大小重组为 ROI 矩阵。

        把变长的一阶段输出对齐到 (batch_size, NMS_POST_MAXSIZE, ...) 的固定张量，
        写入 example 的 rois / roi_labels / roi_scores / roi_features 字段。

        Args:
            first_pred (list): 一阶段预测结果（每样本一个 dict）。
            example (dict): 输入数据，结果写回其中。
            features (list): 第二阶段采样得到的特征（流数量 x batch）。

        Returns:
            example: 写入 ROI 字段后的输入字典。
        """
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1] 
        feature_vector_length = sum([feat[0].shape[-1] for feat in features])

        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, box_length 
        ))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE
        ))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size,
            self.NMS_POST_MAXSIZE), dtype=torch.long
        )
        roi_features = features[0][0].new_zeros((batch_size, 
            self.NMS_POST_MAXSIZE, feature_vector_length 
        ))

        for i in range(batch_size):
            num_obj = features[0][i].shape[0]
            # 将 rotation_y 移到第 6 位，使 box 布局变为 7 + C；C 在 nuScenes 中为 2（速度）。

            box_preds = first_pred[i]['box3d_lidar']

            if self.roi_head.code_size == 9:
                # code_size=9 的 box 顺序为 x,y,z,w,l,h,rotation_y,velocity_x,velocity_y，此处调整为统一顺序。
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']
            roi_features[i, :num_obj] = torch.cat([feat[i] for feat in features], dim=-1)

        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores  
        example['roi_features'] = roi_features

        example['has_class_labels']= True 

        return example 

    def post_process(self, batch_dict):
        """将 roi_head 输出解码为最终预测框。

        对每帧：还原旋转位置、融合第二阶段 IoU 分数与一阶段分数、过滤背景标签。

        Args:
            batch_dict (dict): roi_head 的输出（含 batch_box_preds/batch_cls_preds 等）。

        Returns:
            list: 每帧的预测字典（box3d_lidar/scores/label_preds/metadata）。
        """
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # 这里 cls_preds 是第二阶段预测的 IoU。 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # 将 rotation 移回末尾（生成提交文件时取 0:6 列与 -1 列）。
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            # 融合第二阶段 IoU（经 sigmoid）与一阶段分数：取二者几何平均。
            scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
            # 过滤背景（label==0）。
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # 当前推理不需要 NMS。 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 


    def forward(self, example, return_loss=True, **kwargs):
        """两阶段前向：一阶段预测 → 第二阶段采样 → roi_head 精炼。

        Args:
            example: 输入数据。
            return_loss (bool): True 时返回合并损失。

        Returns:
            训练返回合并损失字典；推理返回后处理后的预测列表。
        """
        out = self.single_det.forward_two_stage(example, 
            return_loss, **kwargs)

        if len(out) == 5:
            one_stage_pred, bev_feature, voxel_feature, final_feature, one_stage_loss = out 
            example['voxel_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        # N C H W -> N H W C：BEV 特征转为 NHWC，便于第二阶段做点级采样。 
        if self.use_final_feature:
            example['bev_feature'] = final_feature.permute(0, 2, 3, 1).contiguous()
        else:
            example['bev_feature'] = bev_feature.permute(0, 2, 3, 1).contiguous()
        
        centers_vehicle_frame = self.get_box_center(one_stage_pred)

        if self.roi_head.code_size == 7 and return_loss is True:
            # code_size=7 时丢弃速度维，使 GT 与预测维度保持一致。 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        features = [] 

        for module in self.second_stage:
            feature = module.forward(example, centers_vehicle_frame, self.num_point)
            features.append(feature)
            # feature 为两层列表：第一层是第二阶段信息流数量，第二层是 batch。 

        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example, features=features)

        # 最终分类 / 回归阶段（roi_head）。 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)
