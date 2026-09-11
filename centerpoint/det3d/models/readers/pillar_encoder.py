"""PointPillars 柱特征编码与散布模块（源自 SECOND 的实现）。

由 Alex Lang 与 Oscar Beijbom 于 2018 年编写，MIT 许可证 [见 LICENSE]。

将体素化的点云（柱）编码为固定维度特征（PillarFeatureNet），再散布回 BEV 伪图像
（PointPillarsScatter）。这是 CenterPoint 论文 Sec 3 骨架中 reader（柱编码）与
backbone（柱散布）部分。

主要类：
    - PFNLayer: 单层柱特征网络，逐点 MLP + max 池化（非末层时拼接回特征）。
    - PillarFeatureNet: 柱特征网络，输出每个柱的特征向量。
    - PointPillarsScatter: 将柱特征散布回 BEV 伪图像。
"""

import torch
from det3d.models.utils import get_paddings_indicator
from torch import nn
from torch.nn import functional as F
from ..registry import BACKBONES, READERS
from ..utils import build_norm_layer


class PFNLayer(nn.Module):
    """单层柱特征网络（Pillar Feature Net Layer）。

    对柱内每个点经线性层 + 归一化 + ReLU 映射到新特征，再在柱内做 max 池化；
    非末层把池化结果拼回逐点特征，末层只返回池化结果。
    """

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        """构造 PFNLayer。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数（非末层会被减半作为拼接后的宽度）。
            norm_cfg (dict, optional): 归一化层配置。
            last_layer (bool): 是否为最后一层；末层不再拼接池化特征。
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        """逐点映射 → 归一化 → 激活 → 柱内 max 池化（可选拼接）。

        Args:
            inputs (Tensor): (P, num_points, C) 柱内点特征。

        Returns:
            Tensor: 末层返回 (P, 1, C') 的池化特征；否则返回拼接后的逐点特征。
        """

        x = self.linear(inputs)
        # 归一化层期望 (N, C, L) 布局，故先转置到通道维，归一化后再转回。
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        # 柱内最大池化（维度 1 为每柱的点）。
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            # 非末层：把池化特征广播回每点并沿通道拼接，实现特征聚合。
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@READERS.register_module
class PillarFeatureNet(nn.Module):
    """柱特征网络：对柱内点补充装饰特征后经 PFNLayer 编码为固定维度特征。"""

    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
        virtual=False
    ):
        """构造柱特征网络。

        Args:
            num_input_features (int): 柱内点的输入特征维度（xyz 或 xyz+intensity）。
            num_filters (tuple): 各 PFNLayer 的输出通道数。
            with_distance (bool): 是否附加点到原点的欧氏距离特征。
            voxel_size (tuple): 体素（柱）尺寸，仅使用前两维。
            pc_range (tuple): 点云范围，仅使用起点的 x / y 坐标。
            norm_cfg (dict, optional): 归一化层配置。
            virtual (bool): 是否使用虚拟点处理。
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input = num_input_features
        # 额外拼接 5 维装饰特征（相对柱均值 3 维 + 相对柱中心 2 维）；可选再拼接距离特征。
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # 构造多个 PFNLayer。
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.virtual = virtual 

        # 记录柱尺寸与 x/y 方向偏移，用于计算点到柱中心的相对坐标。
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        """为柱内点补充装饰特征，经 PFNLayer 编码得到每个柱的特征。

        Args:
            features (Tensor): (P, max_points, C) 柱内点特征。
            num_voxels (Tensor): (P,) 每个柱内的真实点数。
            coors (Tensor): (P, 4) 柱坐标（batch, z, y, x）。

        Returns:
            Tensor: 编码后的柱特征（经 squeeze 去除点数维）。
        """
        device = features.device

        if self.virtual:
            # 虚拟点处理：先临时修正指示位后再参与特征计算。
            virtual_point_mask = features[..., -2] == -1
            virtual_points = features[virtual_point_mask]
            virtual_points[..., -2] = 1
            features[..., -2] = 0 
            features[virtual_point_mask] = virtual_points

        dtype = features.dtype
        # 计算点相对柱均值（cluster center）的偏移。
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # 计算点相对柱中心在 x/y 方向的偏移。
        # f_center = features[:, :, :2]
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # 拼接原始特征与装饰特征。
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # 装饰特征未区分空柱，这里用掩码把空柱（padding 点）特征置零。
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # 逐层通过 PFNLayer。
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@BACKBONES.register_module
class PointPillarsScatter(nn.Module):
    """柱散布层：将学到的柱特征按坐标散布回稠密 BEV 伪图像。"""

    def __init__(
        self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs
    ):
        """构造柱散布层。

        Args:
            num_input_features (int): 输入（柱）特征通道数。
            norm_cfg (dict, optional): 归一化配置（未使用）。
            name (str): 模块名称。
            **kwargs: 额外参数（忽略）。
        """

        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        """按体素坐标将柱特征散布回 (B, C, H, W) 的 BEV 伪图像。

        Args:
            voxel_features (Tensor): (P, C) 柱特征。
            coords (Tensor): (P, 4) 柱坐标（batch, z, y, x）。
            batch_size (int): batch 大小。
            input_shape: BEV 网格形状 [nx, ny]（即 [W, H]）。

        Returns:
            Tensor: (B, C, ny, nx) 的 BEV 伪图像。
        """
        self.nx = input_shape[0]
        self.ny = input_shape[1]

        # batch_canvas 保存最终输出。
        batch_canvas = []
        for batch_itt in range(batch_size):
            # 为当前样本创建空白画布（按列展平）。
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # 仅处理当前 batch 的非空柱。
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            # 将柱的 (y, x) 网格坐标展平为一维索引：ind = y * nx + x。
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # 将柱特征散布回画布对应位置。
            canvas[:, indices] = voxels

            # 追加到列表，稍后堆叠。
            batch_canvas.append(canvas)

        # 堆叠为 3 维张量 (batch, channels, rows*cols)。
        batch_canvas = torch.stack(batch_canvas, 0)

        # 恢复列展平前的形状，得到 4 维 BEV 特征 (B, C, H, W)。
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas
