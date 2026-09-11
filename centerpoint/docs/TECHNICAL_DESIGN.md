# CenterPoint 白盒技术设计文档（Waymo 单阶段 VoxelNet）

> 本文档是 CenterPoint 的**技术设计文档**，从白盒视角描述模型内部工作原理、数据处理流程与设计决策，作为操作型 [README](../README.md) 的补充。
>
> 参照配置：`configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py`（单帧 `nsweeps=1`、无速度分支、体素 `0.1×0.1×0.15` m、点云范围 `±75.2` m）。所有张量维度与数值均以该配置为基准，除非特别说明。

---

## 目录

1. [文档约定与坐标系](#1-文档约定与坐标系)
2. [整体架构与模块划分](#2-整体架构与模块划分)
3. [端到端数据流总览](#3-端到端数据流总览)
4. [数据预处理层详细设计](#4-数据预处理层详细设计)
5. [网络编码层详细设计](#5-网络编码层详细设计)
6. [检测头与损失层详细设计](#6-检测头与损失层详细设计)
7. [后处理与结果导出层详细设计](#7-后处理与结果导出层详细设计)
8. [关键设计决策说明](#8-关键设计决策说明)

---

## 1. 文档约定与坐标系

### 1.1 单位约定

| 物理量 | 单位 | 说明 |
|---|---|---|
| 坐标 `x, y, z` | 米（meter） | 点云与框中心均为米制 |
| 尺寸 `dx, dy, dz`（内部）/ `length, width, height`（Waymo） | 米 | 均为正数 |
| 朝向角 `yaw` / `heading` | 弧度（radian） | 详见 1.2 |
| 速度 `vx, vy` | 米/秒 | 参考系下的水平速度 |
| `intensity`（反射强度） | 无量纲 | 在线加载时经 `tanh` 归一化到 (-1, 1) |
| `elongation`（脉冲延展） | 无量纲 | Waymo 原始值，不做归一化 |
| `time_lag`（多 sweep 时间差） | 秒 | 历史帧相对当前帧的时间差 |

### 1.2 坐标系与 box 约定

CenterPoint 内部统一使用 **KITTI-lidar 风格**的 box 表示：

```text
[x, y, z, dx, dy, dz, yaw]
```

- `x, y, z`：3D box 中心在 lidar 坐标系下的位置；
- `dx, dy, dz`：box 在 x/y/z 三轴上的尺寸（`dx` 对应 Waymo 的 `width`，`dy` 对应 Waymo 的 `length`）；
- `yaw`：绕 z 轴的旋转角，**从 y 轴负方向逆时针量取**，范围 `[-π, π]`。

Waymo 原始 box 约定为：

```text
[x, y, z, length, width, height, vel_x, vel_y, heading]
```

- `length` 沿物体朝向主轴、`width` 沿其短轴；
- `heading`：**从 x 轴正方向顺时针量取**。

**转换关系**（在 [waymo_common.py](../det3d/datasets/waymo/waymo_common.py) 的 `_fill_infos` 中完成，逆变换在 `_create_pd_detection` 中完成）：

```text
内部 yaw   = -π/2 - heading_waymo
内部 dx    = width_waymo
内部 dy    = length_waymo
内部 dz    = height_waymo
```

> 正向推导：Waymo `heading` 从 +x 轴顺时针量取，内部 `yaw` 从 -y 轴逆时针量取，两者相差 `-π/2`；正/逆变换均统一为 `yaw = -π/2 - heading` 且 `heading = -yaw - π/2`（两者互为自反）。

---

## 2. 整体架构与模块划分

CenterPoint 单阶段（以 VoxelNet 为例）由「数据预处理 → 网络编码 → 检测头 → 后处理」四层构成：

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 数据预处理层（离线 + 在线）                                            │
│  TFRecord → 逐帧 pickle → infos → 点云/标注加载 → 增强 →              │
│  体素化 → 标签分配(AssignLabel) → batch 整理                           │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 网络编码层                                                             │
│  体素特征提取(VFE) → 稀疏卷积 backbone → BEV neck(RPN)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 检测头与损失层                                                           │
│  CenterHead(shared_conv + SepHead) → 中心热图 + 属性回归               │
│  训练: FastFocalLoss + RegLoss；推理: predict 解码                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 后处理与结果导出层                                                       │
│  阈值/范围过滤 → 旋转 NMS → 内部 box 逆变换 → detection_pred.bin        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 模块职责表

| 模块 | 源码位置 | 职责 |
|---|---|---|
| TFRecord 解码 | `datasets/waymo/waymo_converter.py` + `waymo_decoder.py` | 离线把 tfrecord 解析为点云/标注 pickle |
| info 生成 | `datasets/waymo/waymo_common.py` | 生成样本索引 `infos_*.pkl`，完成坐标/src 转换 |
| 点云/标注加载 | `datasets/pipelines/loading.py` | 从 pickle 读取点云与 GT，多 sweep 对齐 |
| 数据增强 | `datasets/pipelines/preprocess.py`（`Preprocess`） | GT 过滤、DB 采样、随机翻转/旋转/缩放/平移 |
| 体素化 | `datasets/pipelines/preprocess.py`（`Voxelization`）+ `core/input/voxel_generator.py` | 点云 → 稀疏体素 |
| 标签分配 | `datasets/pipelines/preprocess.py`（`AssignLabel`） | 构造热图与回归监督目标 |
| 数据整理 | `datasets/pipelines/formating.py`（`Reformat`） | 打包成模型输入 batch |
| 体素编码 | `models/readers/voxel_encoder.py` | 体素内点特征 → 体素特征 |
| 稀疏 backbone | `models/backbones/scn.py`（`SpMiddleResNetFHD`） | 稀疏卷积提取特征 → 稠密 BEV |
| BEV neck | `models/necks/rpn.py`（`RPN`） | 多尺度 BEV 特征融合 |
| 检测头 | `models/bbox_heads/center_head.py`（`CenterHead`） | 中心热图 + 属性回归 + 解码 |
| 损失 | `models/losses/centernet_loss.py` | focal loss + L1 回归损失 |
| 旋转 NMS | `core/bbox/box_torch_ops.py` | 旋转框 NMS |
| 结果导出 | `datasets/waymo/waymo_common.py` | 逆变换为 Waymo protobuf |

---

## 3. 端到端数据流总览

下面以「单帧 Waymo 样本」为主线，列出数据在模块间流转时的张量形状变化（`B`=batch，`N`=点数，`M`=体素数，`K`=类别数/目标数）：

| 阶段 | 输入 | 输出 | 关键维度（本例） |
|---|---|---|---|
| TFRecord 解码 | `.tfrecord` | 点云 pickle | `points_xyz [N,3]`，`points_feature [N,2]` |
| info 生成 | pickle | `infos_*.pkl` | `gt_boxes [K,9]`（内部约定） |
| 点云加载 | info | `points` | `[N,5]` float32 |
| 标注加载 | info | `annotations` | `boxes [K,9]`，`names [K]` |
| 增强 | points/annos | points/annos | `[N,5]` / `boxes [K',9]` |
| 体素化 | points | voxels | `voxels [M,5,5]`，`coords [M,3]`，`num_points [M]` |
| 标签分配 | annos | targets | `hm [3,188,188]`，`anno_box [500,10]`，`ind/mask/cat [500]` |
| 体素编码 | voxels | voxel feats | `[M,5]` |
| backbone | voxel feats+coors | BEV | `[B,256,188,188]` |
| neck | BEV | BEV 融合 | `[B,512,188,188]` |
| 检测头 | BEV 融合 | preds | `hm [B,3,188,188]`，`reg [B,2,…]`，`height [B,1,…]`，`dim [B,3,…]`，`rot [B,2,…]` |
| 解码+NMS | preds | 检测结果 | `box3d_lidar [K,7]`，`scores [K]`，`label_preds [K]` |
| 结果导出 | 检测结果 | protobuf | `detection_pred.bin` |

---

## 4. 数据预处理层详细设计

### 4.1 TFRecord 解码（离线）

**模块**：`waymo_decoder.py` 的 `decode_frame` / `decode_annos`，由 `waymo_converter.py` 调用。

- **输入**：Waymo `dataset_pb2.Frame` protobuf。
- **点云解码**（`extract_points`）：对 TOP 激光的第一/第二回波 range image 解压，结合外参、波束倾角、像素位姿做反投影，得到笛卡尔点云：
  - `points_xyz`：`[N, 3]` float32（米）；
  - `points_feature`：`[N, 2]` float32，依次为 `intensity`、`elongation`。
- **标注解码**（`extract_objects`）：每个目标输出：
  - `box`：`[x, y, z, length, width, height, vel_x, vel_y, heading]` float32（速度已由全局系转到参考系平面）；
  - `name`、`num_points`、`combined_difficulty_level` 等。
- **输出**：`lidar/seq_{i}_frame_{j}.pkl` 与 `annos/seq_{i}_frame_{j}.pkl`。

### 4.2 info 生成与坐标转换（离线）

**模块**：`waymo_common.py` 的 `_fill_infos`。

- **输入**：上述逐帧 pickle。
- **处理**：
  1. 记录当前帧 `path` 与 `anno_path`；
  2. 读取 GT 并过滤 `num_points == 0` 的目标；
  3. 将 Waymo box 转为内部约定（§1.2）；
  4. 多 sweep 时记录历史帧 `path`、`transform_matrix`（`[4,4]`）、`time_lag`。
- **输出**：`infos_{split}_{NN}sweeps_filter_zero_gt.pkl`，每个样本为 dict，键包含 `token`、`path`、`anno_path`、`gt_boxes [K,9]`、`gt_names`、`sweeps`（多 sweep）等。

### 4.3 点云加载（在线）

**模块**：`loading.py` 的 `LoadPointCloudFromFile` 与辅助函数 `read_single_waymo`。

- **输入**：info 中的 `path`。
- **算法**：
  1. 读取点云 pickle，得到 `points_xyz [N,3]`、`points_feature [N,2]`；
  2. `intensity` 做 `tanh` 归一化；
  3. 拼接为 `points [N,5] = [x, y, z, tanh(intensity), elongation]`。
- **多 sweep**（`nsweeps>1`）：历史帧经 `transform_matrix` 齐次变换对齐到当前帧，并把 `time_lag` 拼到特征尾部，得到 `combined [N',6]`。
- **输出**：`res["lidar"]["points"]`（`[N,5]`）与 `res["lidar"]["combined"]`（多 sweep 时 `[N',6]`）。

### 4.4 标注加载（在线）

**模块**：`loading.py` 的 `LoadPointCloudAnnotations`。

- **输入**：info 中的 `gt_boxes`、`gt_names`。
- **输出**：`res["lidar"]["annotations"] = {"boxes": [K,9] float32, "names": [K]}`（Waymo 不含速度字段名，速度已内嵌在 box 的第 6:8 维）。

### 4.5 数据增强（在线，仅训练）

**模块**：`preprocess.py` 的 `Preprocess`。

- **输入**：`points [N,5]`、`annotations`、`metadata`。
- **处理顺序**：
  1. 过滤 `DontCare/ignore/UNKNOWN` GT；
  2. （若启用 DB 采样）做 GT copy-paste 采样以缓解长尾；
  3. 类别映射为 `class_names.index(name) + 1`（`0` 留给背景）；
  4. 随机翻转 → 全局旋转（`[-0.785, 0.785]` rad）→ 缩放（`[0.95, 1.05]`）→ 平移（std=0.5 m），点云与 box 同步变换。
- **输出**：增强后的 `points [N,5]` 与 `annotations`（含 `gt_classes`）。

### 4.6 体素化（在线）

**模块**：`preprocess.py` 的 `Voxelization` + `core/input/voxel_generator.py` 的 `VoxelGenerator`。

- **配置**：`voxel_size=[0.1,0.1,0.15]`，`pc_range=[-75.2,-75.2,-2,75.2,75.2,4]`，`max_points_in_voxel=5`，`max_voxel_num=[150000,200000]`。
- **算法**：由点云范围与体素尺寸反推网格 `grid_size = (range_max - range_min) / voxel_size = [1504, 1504, 40]`；底层 CUDA 算子 `points_to_voxel` 把点分配到体素。
- **输出**（写入 `res["lidar"]["voxels"]`）：
  - `voxels`：`[M, 5, 5]` float32（每体素最多 5 个点，每点 5 维特征，不足补零）；
  - `coordinates`：`[M, 3]` 整型体素坐标；
  - `num_points`：`[M]` 每体素真实点数；
  - `num_voxels`：`[1]`；
  - `shape/range/size`：体素网格形状/范围/尺寸（供下游计算 feature map）。

### 4.7 标签分配（在线，核心）

**模块**：`preprocess.py` 的 `AssignLabel`（对应论文 Sec 3.1/3.2）。

- **输入**：增强后的 `annotations`、体素 `shape/range/size`。
- **feature map 尺寸**：`feature_map_size = grid_size[:2] // out_size_factor = 1504 // 8 = [188, 188]`。
- **算法**（对每个 task、每个 GT 目标）：
  1. 中心投影到特征网格：`coor_x = (x - pc_range[0]) / voxel_size[0] / out_size_factor`；
  2. 由 BEV 尺寸计算高斯半径 `radius = max(min_radius, gaussian_radius((l,w), min_overlap))`；
  3. 在热图对应类别通道上以 `ct=(coor_x, coor_y)` 为中心绘制 2D 高斯；
  4. 记录展平中心索引 `ind = ct_int_y * W + ct_int_x`；
  5. 构造回归目标 `anno_box`（Waymo 共 10 维）：
     ```text
     [offset_x, offset_y, z, log(dx), log(dy), log(dz), vx, vy, sin(yaw), cos(yaw)]
     ```
     其中 `offset = ct - ct_int` 为亚像素小数偏移。
- **输出**（`res["lidar"]["targets"]`，按 task 组织）：
  - `hm`：`[K, 188, 188]` float32（每类别一张中心热图）；
  - `anno_box`：`[500, 10]` float32（`max_objs=500`）；
  - `ind`：`[500]` int64；
  - `mask`：`[500]` uint8（1 表示该槽位有效）；
  - `cat`：`[500]` int64（类别 id）；
  - `gt_boxes_and_cls`：`[500, 10]`（供两阶段精炼使用）。

### 4.8 数据整理（在线）

**模块**：`formating.py` 的 `Reformat`。

- **输入**：`res["lidar"]` 中累积的 `points/voxels/targets`。
- **输出**：`data_bundle` dict，训练模式含 `voxels/coordinates/num_points/num_voxels/shape` 与 `targets`；评测模式含 `points/voxels`，若开启 double_flip 则返回 4 个翻转版本（原始/y 翻转/x 翻转/双翻转）供 TTA。
- batch 由 dataloader 的 collate 按 `DataContainer` 规则堆叠为 `[B, ...]`。

---

## 5. 网络编码层详细设计

### 5.1 体素特征提取（reader）

**模块**：`VoxelFeatureExtractorV3`（`voxel_encoder.py`）。

- **输入**：`voxels [M,5,5]`、`num_points [M]`、`num_voxels`。
- **算法**：对每个体素内点特征求和后除以真实点数（`padding` 点为 0 不影响求和），得到体素均值特征。
- **输出**：`voxel_features [M,5]`（每个体素一个 5 维特征向量）。
- **设计说明**：这是最简的体素编码（点均值，无 PointNet/VFE 扩展），论文 §3 指出 CenterPoint 与任意 3D 编码器兼容。

### 5.2 稀疏卷积 backbone

**模块**：`SpMiddleResNetFHD`（`scn.py`）。

- **输入**：`voxel_features [M,5]`、`coors [M,4]`（`(batch, z, y, x)`）、`batch_size`、`input_shape`（体素网格 `[D,H,W]=[40,1504,1504]`）。
- **结构**（稀疏卷积，通道数 → 每级空间下采样）：

  | 层 | 算子 | 通道 | 空间变化（x/y） |
  |---|---|---|---|
  | conv_input | SubMConv3d | 5 → 16 | 不变 |
  | conv1 | 2×SparseBasicBlock | 16 → 16 | 不变 |
  | conv2 | SparseConv3d stride2 | 16 → 32 | /2 |
  | conv3 | SparseConv3d stride2 | 32 → 64 | /2（累计 /4） |
  | conv4 | SparseConv3d stride2 | 64 → 128 | /2（累计 /8） |
  | extra_conv | SparseConv3d `(3,1,1)` stride `(2,1,1)` | 128 → 128 | 仅 Z 压缩 |

- **算法**：构造 `SparseConvTensor` 做多级稀疏卷积；`extra_conv` 把 Z 维压缩后，`dense()` 转稠密，再把 Z 维折叠进通道维，得到 BEV。
- **输出**：
  - 稠密 BEV：`[B, 256, 188, 188]`（`256 = 128 通道 × Z 压缩后 2`；`188 = 1504/8`，即 `ds_factor=8`）；
  - `multi_scale_voxel_features`：`{conv1..conv4}` 稀疏特征（供两阶段精炼采样）。

### 5.3 BEV neck（RPN）

**模块**：`RPN`（`rpn.py`）。

- **配置**：`layer_nums=[5,5]`、`ds_layer_strides=[1,2]`、`ds_num_filters=[128,256]`、`us_layer_strides=[1,2]`、`us_num_filters=[256,256]`、`num_input_features=256`。
- **输入**：`[B,256,188,188]`。
- **算法**：两级下采样卷积块；每级输出接反卷积上采样到统一分辨率后，在通道维拼接。

  | 块 | 输入通道 | 输出通道 | 下采样 | 上采样分支输出 |
  |---|---|---|---|---|
  | block0 | 256 | 128 | stride 1 | 256（stride 1 卷积） |
  | block1 | 128 | 256 | stride 2 | 256（stride 2 转置卷积） |

- **输出**：`[B, 512, 188, 188]`（`512 = 256 + 256` 通道拼接；空间分辨率经下采样再上采样后保持 `188×188`）。
- **下采样倍率语义**：`downsample_factor = prod(ds_strides) / us_strides[-1]`，与 backbone 的 `ds_factor=8` 联合决定最终特征图分辨率。

---

## 6. 检测头与损失层详细设计

### 6.1 检测头（CenterHead）

**模块**：`CenterHead`（`center_head.py`）。

- **输入**：`[B,512,188,188]`。
- **共享卷积**：`shared_conv`（Conv 512→64 + BN + ReLU）→ `[B,64,188,188]`。
- **SepHead**（每 task 一个，本例单 task，3 类）：

  | 分支 | 输出通道 | 语义 |
  |---|---|---|
  | `hm` | 3 | 每类一张中心热图（logits，后续 sigmoid） |
  | `reg` | 2 | 中心亚像素偏移 (dx, dy) |
  | `height` | 1 | 中心高度 z |
  | `dim` | 3 | 尺寸 log(dx),log(dy),log(dz) |
  | `rot` | 2 | 朝向 (sin yaw, cos yaw) |

  `hm` 分支输出卷积 bias 初始化为 `-2.19`（使 sigmoid 初值 ≈0.1，缓解 focal loss 早期负样本压力）；回归分支用 kaiming 初始化。

- **输出**：`preds_dicts = [ {hm, reg, height, dim, rot} ]`，每张 `[B, C, 188, 188]`；同时返回共享特征 `x [B,64,188,188]`。

### 6.2 损失计算（训练）

**模块**：`CenterHead.loss` + `centernet_loss.py`。

**回归目标拼接**（把各分支在通道维复原为 `anno_box`）：

```text
无 vel: cat(reg[2], height[1], dim[3], rot[2]) → 8 维
有 vel: cat(reg[2], height[1], dim[3], vel[2], rot[2]) → 10 维
```

Waymo 本例无 `vel` 分支，故 GT `anno_box` 从 10 维中剔除速度通道（取索引 `[0,1,2,3,4,5,-2,-1]`）得到 8 维。

**热图损失（FastFocalLoss，论文公式(2)）**：
- `hm` 先过 sigmoid 并截断到 `[1e-4, 1-1e-4]`；
- `gt = (1 - Y)^4` 为 penalty-reduced 惩罚项；
- 负样本损失在全图求和：`-(1-p)^2 · p^2 · gt · log(1-p)` 的等价形式；
- 正样本损失仅通过 `ind` gather：`-log(p)·(1-p)^2`；
- 总损失按正样本数归一化。

**回归损失（RegLoss，论文公式(3)）**：
- 用 `_transpose_and_gather_feat` 在 `ind` 位置 gather 预测，`mask` 屏蔽无效目标；
- 计算 masked L1，按 `code_weights` 加权求和。

**总损失**：

```text
total_loss = hm_loss + weight(=2) · loc_loss
```

**输出**：`{loss, hm_loss, loc_loss, loc_loss_elem, num_positive}`，按 task 聚合。

### 6.3 预测解码（推理）

**模块**：`CenterHead.predict`。

对每个 task、每个像素：

```text
hm   = sigmoid(hm)                         # 中心置信度
dim  = exp(dim)                            # 恢复真实尺寸
yaw  = atan2(rot_sin, rot_cos)             # 恢复朝向角
xs   = (grid_x + reg_x) · out_size_factor · voxel_size_x + pc_range_x
ys   = (grid_y + reg_y) · out_size_factor · voxel_size_y + pc_range_y
```

其中 `out_size_factor=8`、`voxel_size=[0.1,0.1]`、`pc_range=[-75.2,-75.2]`，把特征网格坐标还原为物理米制坐标。

- **无 vel 分支**时最终框：`[x, y, z, dx, dy, dz, yaw]`（7 维）；
- **有 vel 分支**时：[x, y, z, dx, dy, dz, vx, vy, yaw]（9 维）。
- 可选 `double_flip`：把 4 个翻转预测还原到原坐标系后取平均（TTA）。

---

## 7. 后处理与结果导出层详细设计

### 7.1 后处理（NMS）

**模块**：`CenterHead.post_processing` + `box_torch_ops.rotate_nms_pcdet`。

对每个样本：

1. 每个像素取 `K` 类中置信度最大的类别：`scores, labels = max(hm, dim=-1)`；
2. `score_threshold=0.1` 过滤低置信度；
3. `post_center_limit_range=[-80,-80,-10,80,80,10]` 范围过滤中心点；
4. 旋转 NMS（`nms_iou_threshold=0.7`，`nms_pre_max_size=4096`，`nms_post_max_size=500`），内部调用 CUDA 内核 `iou3d_nms_cuda`。

**输出**：每个样本 `{box3d_lidar [K,7], scores [K], label_preds [K]}`。

### 7.2 结果导出（逆变换）

**模块**：`waymo_common.py` 的 `_create_pd_detection`。

- **输入**：检测结果 dict（key 为 frame token）。
- **逆变换**：
  ```text
  yaw → heading:  heading = -yaw - π/2
  尺寸交换:       [dx, dy] → [width, length]（即内部第 4、5 维交换回 Waymo 的 length/width）
  ```
- **输出**：`detection_pred.bin`（Waymo `metrics_pb2.Objects` protobuf）。官方 mAP/mAPH 由 Waymo Open Dataset devkit 计算。

---

## 8. 关键设计决策说明

1. **中心点表示替代 anchor**：3D 物体朝向任意，轴对齐 anchor 难以枚举旋转方向；中心点无固有方向，缩小搜索空间（论文 Sec 1）。
2. **热图用 2D 高斯软化**：中心预测允许少量栅格偏差，`gaussian_overlap=0.1` 与 `min_radius=2` 控制正向监督范围；相比 one-hot 更稳定。
3. **回归目标编码**：尺寸取 `log` 缩小尺度差异、朝向用 `(sin, cos)` 避免 `±π` 不连续、中心用小数偏移 `reg` 补偿下采样量化误差（论文 Sec 3.2）。
4. **仅在 GT 中心回归**：回归输出是密集型，但损失只在 `ind` 位置 gather 计算，大幅降低显存（论文 Sec 3.3）。
5. **速度与跟踪**：速度分支预测 `(vx,vy)`，跟踪退化为贪心最近点匹配（论文 Sec 3.5）；Waymo 本例 `nsweeps=1` 无速度分支。
6. **两阶段精炼**：`extra_conv` 保留的多尺度稀疏特征 + BEV 特征被 `bird_eye_view` 采样，供 `RoIHead` 做 IoU 引导置信度与 box 精炼（论文 Sec 3.4），本文档以单阶段为主线。
7. **`intensity` 的 `tanh` 归一化**：压缩极端反射强度，避免直接输入的数值范围影响训练稳定性。
8. **`hm` bias 初始化为 -2.19**：使初始 `sigmoid(hm)≈0.1`，缓解 focal loss 训练初期的负样本主导问题。