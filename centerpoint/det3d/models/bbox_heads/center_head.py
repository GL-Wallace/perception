# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Copyright (c) 2019 朱本金
# Licensed under the MIT License
# ------------------------------------------------------------------------------
"""CenterPoint 的检测头模块（CVPR 2021《Center-based 3D Object Detection and Tracking》）。

将 backbone 输出的 BEV 特征图转换为每个类别的中心热图(hm)与各属性回归量。
每个 task 使用独立的 SepHead 预测热图及 reg/height/dim/rot（可选 vel）等回归分支，
训练时调用 loss 计算监督损失，推理时调用 predict 完成解码（热图置信度、尺寸恢复、
朝向恢复、网格坐标还原）与 NMS 后处理。

主要类：
    - FeatureAdaption: 基于 DCN v1 的可变形特征对齐模块，供 DCNSepHead 使用。
    - SepHead: 单任务的分离式检测头，各分支共享输入、独立卷积输出。
    - DCNSepHead: 使用可变形卷积进行特征对齐的分离式检测头。
    - CenterHead: 多任务检测头，组织多个 SepHead 并负责 loss 与 predict/post_processing。

设计思路：
    论文 Sec 3.1：backbone 输出 BEV 特征后，CenterHead 为每类预测一张中心热图，
    GT 中心经下采样映射后以 2D 高斯软化（见 center_utils.draw_umich_gaussian）。
    论文 Sec 3.2：回归目标为 offsets(2) + height(1) + dim(3) + yaw(sin,cos)(2)，
    可选 velocity(2)，对应 SepHead 的 reg/height/dim/rot 分支。
    论文 Sec 3.3：热图用 penalty-reduced focal loss，回归用仅在 GT 中心处的 L1 loss
    （见 det3d.models.losses.centernet_loss）。

与其他模块的关系：
    - 通过 HEADS 注册表注册，由 det3d/models 下的检测器（如 VoxelNet、PointPillars 变体）
      实例化并调用 forward，再依次调用 loss（训练）或 predict+post_processing（推理）。
    - 依赖 det3d.models.losses.centernet_loss 的 FastFocalLoss/RegLoss，
      依赖 det3d.core.box_torch_ops.rotate_nms_pcdet 完成旋转框 NMS。
"""

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.models.utils import Sequential
from ..registry import HEADS
import copy 
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms

class FeatureAdaption(nn.Module):
    """基于 DCN v1 的特征对齐模块，供 DCNSepHead 使用。

    与常规 DCN 不同，其偏移量由 anchor 形状预测而非原始 feature map 得出，
    用于让中心点特征自适应地对齐到物体区域。

    Args:
        in_channels (int): 输入特征通道数。
        out_channels (int): 输出特征通道数。
        kernel_size (int): 可变形卷积核尺寸。
        deformable_groups (int): 可变形卷积分组数。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        # 初始时偏移权重置零，使可变形卷积退化为普通卷积，便于训练初期稳定收敛。
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        # 先由 1x1 卷积预测每个采样点的 (dx, dy) 偏移，再做带偏移的卷积并激活。
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x

class SepHead(nn.Module):
    """分离式检测头：一个 task 的所有预测分支共享输入，各分支独立卷积输出。

    论文 Sec 3.1 的 hm 分支输出 K 通道热图（每类一张）；Sec 3.2 的回归分支
    默认包含 reg(2 偏移)/height(1)/dim(3 log 尺寸)/rot(2 即 sin,cos yaw)，
    可选 vel(2 速度)。hm 分支 bias 初始化为 init_bias（默认 -2.19），使
    sigmoid 初值接近 0.1，缓解 focal loss 训练初期的负样本压力；回归分支用
    kaiming 初始化。

    Args:
        in_channels (int): 输入特征通道数。
        heads (dict): 分支名到 (输出通道数, 卷积层数) 的映射，例如
            {'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2)}。
        head_conv (int): 中间卷积输出通道数。
        final_kernel (int): 输出卷积核尺寸。
        bn (bool): 中间卷积后是否接 BatchNorm。
        init_bias (float): hm 分支输出卷积的初始 bias。
    """
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head in self.heads:
            classes, num_conv = self.heads[head]

            # 每个分支先堆叠 (num_conv-1) 层共享卷积提取特征，再接一层输出卷积。
            # 注意这里是逐分支独立卷积，分支之间不共享权重。
            fc = Sequential()
            for i in range(num_conv-1):
                fc.add(nn.Conv2d(in_channels, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    

            if 'hm' in head:
                # 论文 Sec 3.1：热图分支用固定 bias 初始化，避免初始时前景概率过高。
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        """对每个分支独立前向。

        Args:
            x (Tensor): 共享卷积后的 BEV 特征，shape 为 (B, C, H, W)。

        Returns:
            dict[str, Tensor]: 分支名到预测张量的映射，每个预测 shape 为 (B, out_c, H, W)。
        """
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    """使用可变形卷积进行特征对齐的分离式检测头。

    分类(hm)与回归各自使用独立的 FeatureAdaption 做特征对齐，再分别卷积；
    hm 由专属 cls_head 输出，其余回归分支复用 SepHead(task_head)。

    Args:
        in_channels (int): 输入特征通道数。
        num_cls (int): 该 task 的类别数（hm 输出通道数 K）。
        heads (dict): 回归分支配置，同 SepHead。
        head_conv (int): 中间卷积输出通道数。
        final_kernel (int): 输出卷积核尺寸。
        bn (bool): 是否使用 BatchNorm。
        init_bias (float): hm 分支输出卷积的初始 bias。
    """
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # 分类与回归各自使用独立的可变形特征对齐，避免两类任务共享特征互相干扰。
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # 热图预测分支（论文 Sec 3.1），输出 K 通道中心热图。
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # 其余回归目标（reg/dim/rot 等，论文 Sec 3.2）沿用 SepHead。
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        """分别对齐分类/回归特征后前向。

        Args:
            x (Tensor): 共享卷积后的 BEV 特征，shape 为 (B, C, H, W)。

        Returns:
            dict[str, Tensor]: 预测字典，包含 'hm' 及各回归分支。
        """
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@HEADS.register_module
class CenterHead(nn.Module):
    """CenterPoint 的多任务检测头，组织多个 SepHead 并完成训练损失与推理。

    所有 task 先经过一个共享卷积(shared_conv)降通道，随后每个 task 用独立的
    SepHead/DCNSepHead 预测热图(hm)与回归量(reg/height/dim/rot，可选 vel)。
    训练时调用 loss 计算 focal loss 与 L1 回归损失；推理时调用 predict 完成
    解码（sigmoid 热图、exp 尺寸、atan2 朝向、网格坐标还原）与 NMS。

    Args:
        in_channels (List[int]): 输入通道数，取第一项作为共享卷积输入通道。
        tasks (List): task 配置列表，每个元素含 class_names 等字段。
        dataset (str): 数据集名，影响回归目标拼接方式（如 'nuscenes'/'waymo'）。
        weight (float): 回归损失相对热图损失的权重，总损失 = hm_loss + weight * loc_loss。
        code_weights (List[float]): 各回归量 L1 损失的权重系数。
        common_heads (dict): 各 task 共享的回归分支配置（不含 hm）。
        init_bias (float): hm 分支初始 bias。
        share_conv_channel (int): 共享卷积输出通道数。
        num_hm_conv (int): hm 分支卷积层数。
        dcn_head (bool): 是否使用 DCNSepHead。

    注意:
        box_n_dim 为 9（含 velocity）或 7（不含 velocity），标识单个框回归量的维度数。
    """
    def __init__(
        self,
        in_channels=[128,],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights 
        self.weight = weight  # 回归损失(loc loss)相对热图损失(hm loss)的权重
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        # 回归量维度：含 vel 为 9，不含为 7（reg2+height1+dim3+rot2(+vel2)）。
        self.box_n_dim = 9 if 'vel' in common_heads else 7  
        self.use_direction_classifier = False 

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )

        # 所有 task 共享的卷积：先降通道，各 task 再在此共享特征上预测。
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True)
        )

        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                # hm 为 K 通道热图分支（论文 Sec 3.1），在共享回归分支之外单独加入。
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        """前向传播，返回各 task 的预测字典与共享特征。

        Args:
            x (Tensor): backbone 输出的 BEV 特征，shape 为 (B, C, H, W)。

        Returns:
            Tuple[List[dict], Tensor]: (各 task 预测字典列表, 共享卷积后的特征)。
                两阶段方法可用共享特征继续采样精炼。
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts, x

    def _sigmoid(self, x):
        # 截断到 [1e-4, 1-1e-4]，避免 focal loss 中对 0/1 取 log 导致数值不稳定。
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        """计算一批数据的检测损失（论文 Sec 3.3）。

        对每个 task：
        1) 热图先过 sigmoid（截断保证数值稳定），再用 FastFocalLoss 计算
           penalty-reduced focal loss（论文公式(2)）；
        2) 将各回归分支在通道维拼接为 anno_box，并按数据集约定对齐 GT 回归目标的通道顺序；
        3) 用 RegLoss 仅在 GT 中心位置（ind/mask 指定）计算 L1 回归损失，
           按 code_weights 加权求和得到 loc_loss；
        4) 总损失 = hm_loss + weight * loc_loss。

        Args:
            example (dict): 预处理产出的 GT 字段，按 task 组织，含 hm/anno_box/ind/mask/cat。
            preds_dicts (List[dict]): 每个 task 的预测字典。
            test_cfg (dict): 配置（本方法内未直接使用）。

        Returns:
            dict[str, List]: 以 key 聚合各 task 返回值的统计字典，如
                loss/hm_loss/loc_loss/loc_loss_elem/num_positive。
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # 热图 focal loss（论文 Sec 3.3）：先 sigmoid 成概率再计算损失。
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # 将各回归分支拼回 anno_box，使其通道顺序与 GT 回归目标一致（论文 Sec 3.2）。
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)   
                    # 无 vel 分支时，从 GT 目标中剔除速度通道，保持回归维度对齐。
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                        
            else:
                raise NotImplementedError()

            ret = {}
 
            # 仅在 GT 中心位置的 L1 回归损失（dim/offset/height/rot，可选 vel）。
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        # 将「按 batch 组织」的返回字典转换为「按 key 组织」，便于上游聚合各 task 损失。
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """解码预测并做 NMS，返回最终检测结果；可选支持双翻转(double flip)测试时增强。

        解码流程（论文 Sec 3.2 的逆过程）：
        - hm 过 sigmoid 得到中心置信度；
        - dim 取 exp 恢复真实尺寸（训练时以 log(dim) 为回归目标）；
        - rot 用 atan2(sin, cos) 恢复 yaw 角度；
        - 用特征网格坐标 + reg 预测的小数偏移，再乘以 out_size_factor 与 voxel_size，
          并加上 pc_range 起点，还原物理世界 x/y（论文中下采样倍率 R 的反推）。

        Args:
            example (dict): 推理上下文，含 metadata 等字段。
            preds_dicts (List[dict]): 每个 task 的预测字典。
            test_cfg (dict): 推理配置，含 out_size_factor/voxel_size/pc_range/score_threshold/
                nms/post_center_limit_range 等。

        Returns:
            List[dict]: 每个样本的检测结果，键含 box3d_lidar/scores/label_preds/metadata。
        """
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # 将 NCHW 转为 NHWC，便于按像素展开做逐点解码与后处理。
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # 把翻转后的预测图还原到翻转前的原始坐标系。
                    # 翻转预测按 4 帧一组排列：第 1 帧为原点云，第 2 帧为 X 翻转(y=-y)，
                    # 第 3 帧为 Y 翻转(x=-x)，第 4 帧为 X、Y 双翻转(x=-x, y=-y)。
                    # 注意 torch.flip 定义在高维空间上：dims=[2] 表示沿长度为 H 的轴翻转
                    # （习惯上即 Y 轴），dims=[1] 表示沿 W 轴翻转（习惯上即 X 轴）。
                    # 下面对各帧做反向翻转，把预测图统一回原点云坐标系后再做融合。
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            # 论文 Sec 3.2 的逆过程：热图置信度、尺寸恢复（exp）与朝向解耦。
            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(preds_dict['dim'])

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # 翻转后偏移量需按网格边界取补：y 翻转时 reg_y -> 1 - reg_y。
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # y 翻转：y = -y, theta = pi - theta；
                # sin(pi - theta) = sin(theta), cos(pi - theta) = -cos(theta)，故 cos 取反。
                batch_rotc[:, 1] *= -1

                # x 翻转：x = -x, theta = 2pi - theta；
                # sin(2pi - theta) = -sin(theta), cos(2pi - theta) = cos(theta)，故 sin 取反。
                batch_rots[:, 2] *= -1

                # 双翻转（x、y 同时翻转）。
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            # 由 (sin, cos) 恢复唯一朝向角 yaw（论文 Sec 3.2 的朝向表示）。
            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            # 生成特征网格的整数坐标，配合回归的小数偏移还原亚像素中心。
            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            # 把 feature 网格坐标还原为物理 x/y：乘以下采样倍率与 voxel 尺寸，再加上点云范围起点。
            xs = xs * test_cfg.out_size_factor * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # y 翻转时 vy 取反。
                    batch_vel[:, 1, ..., 1] *= -1
                    # x 翻转时 vx 取反。
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1
                    
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                pass 
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, test_cfg, post_center_range, task_id)) 

        # 合并各 task 的检测结果到同一批样本上。
        ret_list = []
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    # 不同 task 的类别标签前会按累计类别数做偏移，避免标签冲突。
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, test_cfg, post_center_range, task_id):
        """对解码后的预测做阈值过滤与 NMS，得到最终检测框。

        Args:
            batch_box_preds (Tensor): 解码后的框预测 (B, H*W, box_n_dim)。
            batch_hm (Tensor): 解码后的热图置信度 (B, H*W, K)。
            test_cfg (dict): 推理配置，含 score_threshold/nms/min_radius 等。
            post_center_range (Tensor): 中心点合法范围 [x_min,y_min,z_min,x_max,y_max,z_max]。
            task_id (int): 当前 task 编号，用于取用 min_radius 等按 task 配置的参数。

        Returns:
            List[dict]: 每个样本的检测结果，键含 box3d_lidar/scores/label_preds。
        """
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]

            # 每个像素取 K 类中置信度最大的类别作为该位置预测。
            scores, labels = torch.max(hm_preds, dim=-1)

            score_mask = scores > test_cfg.score_threshold
            # 过滤中心点落在合法点云范围之外的预测。
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]

            # 取出 NMS 所需的框字段：中心 x,y,z 与尺寸 w,l,h 及朝向角（最后一个分量）。
            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.get('circular_nms', False):
                # 以中心距离为半径的圆形 NMS（速度更快，适合密集点云）。
                centers = boxes_for_nms[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id], post_max_size=test_cfg.nms.nms_post_max_size)  
            else:
                # 旋转 IoU 的 3D NMS。
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(), 
                                    thresh=test_cfg.nms.nms_iou_threshold,
                                    pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                    post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 

import numpy as np 
def _circle_nms(boxes, min_radius, post_max_size=83):
    """按中心点间距离做 NMS（圆形 NMS）。

    Args:
        boxes (Tensor): (N, 3)，前两列为中心 x/y，第三列为置信度。
        min_radius (float): 中心最小间距阈值，小于该距离的框视为重复。
        post_max_size (int): 最多保留的框数量。

    Returns:
        Tensor: 保留框的下标。
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  