# Waymo 数据转换与数据集逻辑整理

本文按当前 `centerpoint/det3d/datasets/waymo/` 代码实现整理 Waymo Open Dataset 的处理链路：

```text
Waymo TFRecord
  -> waymo_converter.py
     -> waymo_decoder.decode_frame()  生成 lidar/seq_{i}_frame_{j}.pkl
     -> waymo_decoder.decode_annos()  生成 annos/seq_{i}_frame_{j}.pkl
  -> tools/create_data.py waymo_data_prep
     -> waymo_common.create_waymo_infos()
     -> infos_{split}_{nsweeps}sweeps_filter_zero_gt.pkl
     -> train split 额外生成 gt_database / dbinfos
  -> WaymoDataset + pipeline
     -> LoadPointCloudFromFile
     -> LoadPointCloudAnnotations
     -> Preprocess / Voxelization / AssignLabel / Reformat
  -> evaluation()
     -> detection_pred.bin / tracking_pred.bin
     -> 交给 Waymo devkit 计算 mAP/mAPH
```

## 0. 零基础概念补充

这一节保留偏学习笔记式的背景解释，先把 Waymo lidar 数据、坐标系、位姿、回波、RangeImage、标注这些基础概念讲清楚，再去看后面的代码链路会顺很多。

### 0.1 Waymo 5 个激光雷达传感器

Waymo 自车搭载 5 套激光雷达，在 `dataset_pb2.LaserName` 枚举中定义传感器 ID：

| 枚举常量 | 数值 | 安装位置 | 直观理解 |
| --- | ---: | --- | --- |
| `TOP` | 1 | 车顶 | 主激光雷达，覆盖 360 度，是最重要的长距点云来源 |
| `FRONT` | 2 | 车头 | 前向短距 lidar，补充车头附近区域 |
| `SIDE_LEFT` | 3 | 车身左侧 | 左侧短距 lidar，补充侧向盲区 |
| `SIDE_RIGHT` | 4 | 车身右侧 | 右侧短距 lidar，补充侧向盲区 |
| `REAR` | 5 | 车尾 | 后向短距 lidar，补充车尾附近区域 |

代码中会把 `frame.lasers` 和 `frame.context.laser_calibrations` 都按 `name` 排序后 zip 到一起，保证每个 lidar 数据和对应标定一一匹配。

TOP lidar 比较特殊。它是车顶主 lidar，扫描一帧时车还在运动，所以 Waymo 为 TOP 提供了每个 range image 像素对应的 `pixel_pose`，用于补偿运动畸变。非 TOP lidar 在当前实现里没有传入 `pixel_pose`，只用自身外参和 beam inclination 完成点云反投影。

### 0.2 激光雷达为什么会有多回波

一束激光打出去以后，可能不只反射一次：

- **第一回波 first return**：最先返回，通常来自最近的一层物体，比如树叶、网格、车身表面。
- **第二回波 second return**：如果激光穿过了半透明、稀疏或边缘结构，可能在后面的物体上再次反射，形成更晚返回的测量。

实心不透明物体通常只有第一回波；树叶、栏杆、铁丝网、雨雾边缘等场景更可能产生多回波。

Waymo 的 first/second return 不是存在同一张 range image 的某个通道里，而是两个独立 blob：

```text
ri_return1.range_image_compressed
ri_return2.range_image_compressed
```

当前代码会同时读取这两个 return，并把它们 concat 到同一帧点云中。也就是说，训练点云里会包含 first return 和 second return 的有效点，但最终没有额外保留“这个点来自第几回波”的标志位。

### 0.3 RangeImage 是什么

普通点云是 `[N, 3]` 或 `[N, C]` 的散点集合。RangeImage 可以理解为 lidar 的“球面深度图”：每个像素对应一束激光方向，像素值存这束光测到的距离和附加属性。

RangeImage 的 shape 通常是：

```text
H x W x C
```

含义：

- `H`：垂直方向的 beam 数量，可以理解为多少根上下排布的激光线束。
- `W`：水平方向的扫描角度采样数量，可以理解为一圈里分了多少列。
- `C`：每个像素保存的属性通道。

当前代码使用的 Waymo range image 通道：

| 通道索引 | 字段 | 含义 |
| ---: | --- | --- |
| 0 | range | 距离，单位 m；`range <= 0` 代表无效点，会被 mask 丢弃 |
| 1 | intensity | 反射强度，反映回波能量大小；加载训练时会做 `tanh` 归一化 |
| 2 | elongation | 脉冲伸长量，Waymo 提供的附加 lidar 特征 |
| 3 | NLZ 标记 | No Label Zone 相关标记；当前代码解码时临时拼接，但训练不保留 |

容易混淆点：第 3 通道不是 second return。second return 由 `ri_return2` 表示。

### 0.4 NLZ 是什么

NLZ 是 No Label Zone，直译是“无标注区域”。Waymo 数据里某些区域可能没有完整标注，官方会用 NLZ 相关信息帮助评估或过滤。对训练来说，如果把无标注区域里的点当作普通背景，可能会引入噪声，因为那里可能有真实物体但没有标注框。

当前这份 CenterPoint 代码只把 range image 的第 3 通道临时拼进 `[x, y, z, intensity, elongation, nlz]`，随后在 `extract_points()` 里丢弃，只保存 `points_xyz` 和 `points_feature`。因此本仓库默认训练路径没有显式使用 NLZ。

### 0.5 坐标系：lidar、vehicle、global

理解 Waymo 转换时，最重要的是分清三个坐标系：

| 坐标系 | 原点 | 作用 |
| --- | --- | --- |
| lidar 坐标系 | 单个 lidar 自己的光学中心 | RangeImage 根据 range/角度先反投影到这里 |
| vehicle/ego 坐标系 | 自车车体坐标原点 | 多个 lidar 通过外参统一到这里；模型训练也主要在这里 |
| global/world 坐标系 | 场景中的固定世界坐标 | 用于跨帧位姿、多 sweep 对齐、目标速度全局表达 |

常见转换链路：

```text
RangeImage
  -> 球面坐标反投影
  -> lidar 坐标点
  -> lidar extrinsic
  -> vehicle 坐标点
```

如果要跨帧对齐，还会用：

```text
历史帧 vehicle
  -> 历史帧 global
  -> 当前帧 vehicle
```

这就是后面 `tm = ref_from_global @ global_from_car` 的含义。

### 0.6 球面坐标到 lidar 坐标

RangeImage 每个有效像素提供一个距离 `r`。只知道距离还不够，还需要知道这束光的方向。

方向由两个角决定：

| 参数 | 含义 | 来源 |
| --- | --- | --- |
| `beam_inclination` | 垂直方向俯仰角 | lidar calibration，或由 min/max 计算 |
| `azimuth` | 水平方向方位角 | 根据 range image 列索引计算 |

直观上，每个 range image 像素代表“一束朝某个方向发出的光”。有了方向和距离，就能从球面坐标恢复出 lidar 坐标系下的 3D 点。

代码没有手写公式，而是调用 Waymo 官方：

```python
range_image_utils.extract_point_cloud_from_range_image(...)
```

这个函数内部会完成 polar/cartesian 相关转换、外参变换，以及 TOP lidar 的 motion compensation。

### 0.7 外参 extrinsic 是什么

`extrinsic` 是单个 lidar 到 vehicle 坐标系的静态安装变换：

```text
P_vehicle = T_lidar_to_vehicle @ P_lidar
```

它描述 lidar 装在车上的位置和姿态，比如离车体原点多远、朝向相对车体旋转了多少。这个标定在同一段数据中通常固定，不随每帧车辆运动改变。

所以外参解决的是“多个 lidar 怎么统一到同一个车体坐标系”的问题。

### 0.8 frame pose / vehicle pose 是什么

`frame.pose.transform` 是当前帧 vehicle 到 global 的 4x4 齐次变换矩阵：

```text
T_vehicle_to_global
```

它描述当前时刻自车在世界坐标中的位置和朝向。比如车开了一段路，vehicle 坐标系跟着车动，但 global 坐标系不动。每一帧的 `frame.pose` 都可能不同。

它主要用于：

1. 将目标速度从 global 坐标系旋转到当前 vehicle 坐标系。
2. 多 sweep 时把历史帧点云对齐到当前帧坐标系。
3. 如果用户需要，也可以把当前帧 vehicle 点云额外变换到 global 坐标系。

当前训练路径最终使用的是 vehicle 坐标点云，不是 global 坐标点云。

### 0.9 欧拉角和 4x4 齐次矩阵

Waymo 的 TOP `range_image_pose_compressed` 中，每个像素存的是 6 个数：

```text
[roll, pitch, yaw, tx, ty, tz]
```

前三个是姿态角，后三个是平移。

| 参数 | 中文 | 旋转轴 | 直观理解 |
| --- | --- | --- | --- |
| roll | 横滚角 | X 轴 | 车身左右倾斜 |
| pitch | 俯仰角 | Y 轴 | 车头上扬或下俯 |
| yaw | 偏航角 | Z 轴 | 车头在水平面内的朝向 |
| tx/ty/tz | 平移 | 无 | vehicle 原点在 global 中的位置 |

代码不会直接拿欧拉角做点乘，而是先把 roll/pitch/yaw 转成 3x3 旋转矩阵，再和 tx/ty/tz 拼成 4x4 齐次矩阵：

```text
[ R  t ]
[ 0  1 ]
```

齐次矩阵的好处是可以把旋转和平移写成一次矩阵乘法：

```text
P_dst = T_src_to_dst @ [x, y, z, 1]^T
```

### 0.10 Rolling Shutter / motion compensation

TOP lidar 扫一圈不是瞬间完成的。车辆在扫描过程中可能继续前进、转向、颠簸，因此不同像素对应的真实采样时刻略有不同。

如果整张 TOP range image 都用同一个 `frame.pose`，就相当于假设所有激光点在同一瞬间采集，会带来几何拉伸和扭曲。

Waymo 为 TOP 提供 `pixel_pose`，也就是每个 range image 像素自己的采样位姿。代码把这个逐像素 pose 传给官方函数，完成 motion compensation。直观理解：

```text
每个 TOP 像素
  -> 使用该像素采样时刻的位姿
  -> 补偿车身运动造成的畸变
  -> 最终落到统一的 vehicle 坐标表达
```

### 0.11 标注框 box 的含义

Waymo `laser_labels` 中每个 object 有一个 3D box：

```text
center_x, center_y, center_z
length, width, height
heading
```

这些 box 本身已经在当前 vehicle 坐标系下。不要再把它们乘一次 `frame.pose`，否则会错误变到 global 坐标系。

`heading` 是 Waymo 原始朝向角。CenterPoint 内部训练使用自己的 yaw 约定，所以在 infos 阶段会做：

```text
yaw = -pi / 2 - heading
```

并且交换 length/width：

```text
Waymo:      [length, width, height]
CenterPoint [dx=width, dy=length, dz=height]
```

### 0.12 速度为什么只旋转不平移

目标 metadata 中的：

```text
speed_x, speed_y
```

表示 global 坐标系下的速度向量。速度是向量，不是位置点，所以只需要坐标轴旋转，不需要平移。

位置点变换：

```text
P_vehicle = R^-1 @ (P_global - t)
```

速度向量变换：

```text
V_vehicle = R^-1 @ V_global
```

代码中 `global_vel_to_ref()` 做的就是这件事。

### 0.13 frame_name、token、timestamp

`decode_frame()` 和 `decode_annos()` 都会构造：

```text
{scene_name}_{location}_{time_of_day}_{timestamp}
```

但落盘文件名是：

```text
seq_{idx}_frame_{frame_id}.pkl
```

后续 `infos` 中：

- `token` 使用 `seq_{idx}_frame_{frame_id}.pkl`，用于模型预测结果和 info 对齐。
- `timestamp` 从 Waymo 原始 `frame.timestamp_micros` 转成秒。
- 写 Waymo protobuf 评估结果时，再从 annos 里的原始 `frame_name` 取最后的微秒时间戳。

### 0.14 train / val / test 的差异

Waymo 三个 split 的处理入口类似，但标注可用性不同：

- `train`：有标注，生成 infos 后还会创建 GT database。
- `val`：有标注，可用于验证和生成本地 GT protobuf。
- `test`：通常没有可训练 GT，`_fill_infos()` 不写 `gt_boxes/gt_names`。

因此 test split 的 info 主要用于加载点云和输出提交结果，不参与训练标签生成。

### 0.15 一句话区分几个容易混的概念

| 概念 | 解决的问题 |
| --- | --- |
| `extrinsic` | 单个 lidar 坐标如何变到 vehicle 坐标 |
| `frame.pose` | 当前帧 vehicle 坐标如何变到 global 坐标 |
| `pixel_pose` | TOP lidar 每个像素采样时刻的 vehicle/global pose，用于运动补偿 |
| `beam_inclination` | 每一行 range image 对应的垂直激光角度 |
| `ri_return1/ri_return2` | first/second return 两份 range image |
| `RangeImage[..., 3]` | NLZ 相关标记，不是 second return |
| `num_lidar_points_in_box` | Waymo 官方给的框内点数统计，不是当前代码现算 |
| `infos` | 训练数据索引，包含路径、GT、历史 sweep 对齐信息 |
| `gt_database` | 数据增强用的物体级点云库 |

### 0.16 查漏补缺：初学时容易遗漏的边界

1. **TOP 的 `range_image_pose_compressed` 只从 return1 读取**

   当前代码在 TOP lidar 分支中从：

   ```text
   laser.ri_return1.range_image_pose_compressed
   ```

   读取 pixel pose，然后同一份 `pixel_pose` 会用于 first return 和 second return 的点云反投影。不是 return1/return2 各有一套独立 pixel pose。

2. **点云融合发生在 vehicle 坐标系**

   5 个 lidar 各自都有自己的 lidar 坐标系。经过各自 `extrinsic` 后，都会落到同一个 vehicle/ego 坐标系，再 concat 成一帧点云。模型看到的是统一车体坐标下的点，不需要知道点来自哪个 lidar。

3. **当前实现不保留 lidar id**

   虽然原始数据来自 5 个 lidar，但 `extract_points()` 最终只保留 `points_xyz` 和 `points_feature`。训练阶段不知道某个点来自 TOP、FRONT 还是 SIDE lidar。

4. **当前实现不保留 camera 信息**

   Waymo Frame 里还有 camera images、camera labels 等信息，但这个 CenterPoint Waymo 数据路径只处理 lidar 点云和 `laser_labels`。相机 2D 标注不进入当前训练 pipeline。

5. **`frame_id` 和 `timestamp` 不是一回事**

   `frame_id` 是 tfrecord 内部从 0 开始的顺序编号，用于落盘文件名。`timestamp_micros` 是 Waymo 原始微秒时间戳，用于真实时间、`time_lag` 和官方评估结果对齐。

6. **`token` 是仓库内部索引，不是 Waymo 原生 id**

   本仓库 info 里的 `token` 使用 `seq_{idx}_frame_{frame_id}.pkl`。预测结果字典也用这个 token 和 info 对齐。Waymo 官方评估真正需要的是 `context_name` 和 `frame_timestamp_micros`，写 protobuf 时会从 annos pickle 中取回。

7. **`combined_difficulty_level` 和训练过滤不是同一个动作**

   `decode_annos()` 会给 0 点框设置 `combined_difficulty_level = 999`，但真正从训练 `infos` 中移除 0 点框，是 `_fill_infos()` 里的 `mask_not_zero`。如果只看 annos pkl，会看到更多对象；如果看 infos pkl，0 点 GT 已经过滤。

8. **多 sweep 只找同一个 seq 内的历史帧**

   `_fill_infos()` 根据文件名里的 `seq_id` 和 `frame_id` 找 `frame_id - 1`、`frame_id - 2`。序列开头不会跨到上一个 seq，而是用当前帧补齐缺失 sweep。

9. **训练 box 有速度，但模型配置可能不一定使用速度分支**

   Waymo info 中的 `gt_boxes` 是 9 维，含 `vx/vy`。`AssignLabel` 也能构造含速度的 10 维目标。但具体模型是否预测速度，还要看 config 里的 head 配置和 `common_heads`。

10. **评估文件不是指标结果**

    `detection_pred.bin` / `tracking_pred.bin` 只是 Waymo 官方 devkit 的输入文件。本仓库写出 protobuf 后，mAP/mAPH 仍需要额外运行 Waymo 官方评估工具。

## 1. 原始 TFRecord 到逐帧 pickle

入口是 `waymo_converter.py`：

1. `glob(args.record_path)` 找到所有 tfrecord，并排序。
2. 每个 tfrecord 用文件索引 `idx` 作为序列号。
3. tfrecord 内每条 record 解析成 `dataset_pb2.Frame`。
4. 同一帧分别解码点云和标注：
   - `decode_frame(frame, frame_id)` -> lidar pickle
   - `decode_annos(frame, frame_id)` -> annos pickle
5. 输出文件名固定为：
   - `lidar/seq_{idx}_frame_{frame_id}.pkl`
   - `annos/seq_{idx}_frame_{frame_id}.pkl`

推荐目录组织：

```text
WAYMO_DATASET_ROOT
  ├── tfrecord_training/
  ├── tfrecord_validation/
  ├── tfrecord_testing/
  ├── train/
  │   ├── lidar/
  │   └── annos/
  ├── val/
  │   ├── lidar/
  │   └── annos/
  └── test/
      ├── lidar/
      └── annos/
```

典型命令：

```bash
CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
  --record_path 'WAYMO_DATASET_ROOT/tfrecord_training/*.tfrecord' \
  --root_path 'WAYMO_DATASET_ROOT/train/'

CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
  --record_path 'WAYMO_DATASET_ROOT/tfrecord_validation/*.tfrecord' \
  --root_path 'WAYMO_DATASET_ROOT/val/'

CUDA_VISIBLE_DEVICES=-1 python det3d/datasets/waymo/waymo_converter.py \
  --record_path 'WAYMO_DATASET_ROOT/tfrecord_testing/*.tfrecord' \
  --root_path 'WAYMO_DATASET_ROOT/test/'
```

## 2. Waymo 激光雷达与 RangeImage

Waymo 每帧包含 5 个激光雷达：

| LaserName | 数值 | 说明 |
| --- | ---: | --- |
| `TOP` | 1 | 车顶主激光雷达，360 度扫描；代码中使用 `pixel_pose` + `frame_pose` 做 motion compensation |
| `FRONT` | 2 | 前向短距 lidar |
| `SIDE_LEFT` | 3 | 左侧短距 lidar |
| `SIDE_RIGHT` | 4 | 右侧短距 lidar |
| `REAR` | 5 | 后向短距 lidar |

每个 lidar 有两份 range image：

- `ri_return1.range_image_compressed`
- `ri_return2.range_image_compressed`

这两份 blob 分别代表 first return 和 second return。代码会分别解压，然后把两组点 concat 到同一 lidar 点云里。

单个 range image 的通道在当前代码中按如下方式使用：

| 通道 | 字段 | 当前代码用途 |
| ---: | --- | --- |
| 0 | range | `range > 0` 作为有效点 mask |
| 1 | intensity | 后续训练加载时做 `tanh(intensity)` 归一化 |
| 2 | elongation | 作为点特征保留 |
| 3 | NLZ 标记 | 解码时临时拼接，但最终没有返回给训练 |

注意：第 3 通道不是 second-return 标记。first/second return 的区别来自 `ri_return1` / `ri_return2` 两个独立 range image blob。当前实现最终训练点特征不保留 return id，也不保留 NLZ 标记。

## 3. RangeImage 到 vehicle 坐标点云

核心函数是 `waymo_decoder.extract_points_from_range_image()`。

### 3.1 TOP lidar 的 pixel pose

当 `laser.name == LaserName.TOP` 时，代码会额外读取：

```text
laser.ri_return1.range_image_pose_compressed
```

其内容 reshape 后是 `[H, W, 6]`：

```text
[roll, pitch, yaw, tx, ty, tz]
```

代码用 `transform_utils.get_rotation_matrix()` 和 `transform_utils.get_transform()` 将每个像素的 6DoF pose 转成 `[H, W, 4, 4]` 的 `pixel_pose`，并把整帧 `frame.pose.transform` 转成 `frame_pose`。两者一起传给 Waymo 官方的：

```python
range_image_utils.extract_point_cloud_from_range_image(
    range_image,
    extrinsic,
    beam_inclinations,
    pixel_pose=pixel_pose,
    frame_pose=frame_pose,
)
```

这一步负责 TOP lidar 的分时扫描运动补偿。

非 TOP lidar 当前传入：

```python
pixel_pose = None
frame_pose = None
```

即只按 range image、beam inclination 和 lidar extrinsic 反投影到 vehicle 坐标。

### 3.2 beam inclinations

代码优先使用 calibration 内已有的 `beam_inclinations`。如果没有，则根据：

```text
beam_inclination_min
beam_inclination_max
range image height
```

调用 `range_image_utils.compute_inclination()` 生成。

随后必须执行：

```python
beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
```

原因是 range image 行顺序和 beam inclination 的几何顺序相反，不反转会导致点云上下方向错误。

### 3.3 输出点云坐标系

`extract_point_cloud_from_range_image()` 在当前调用方式下输出的是 vehicle/ego 坐标系下的点，而不是世界坐标。

每个 return 临时得到：

```text
[x, y, z, intensity, elongation, nlz]
```

随后 `extract_points()` 会合并：

1. 5 个 lidar，按 `laser.name` 排序后逐个处理。
2. 每个 lidar 的 first return 和 second return。
3. 所有有效点。

最终保存到 lidar pickle 的结构为：

```python
{
    "scene_name": frame.context.name,
    "frame_name": "{scene_name}_{location}_{time_of_day}_{timestamp}",
    "frame_id": frame_id,
    "lidars": {
        "points_xyz": np.ndarray,      # [N, 3], vehicle 坐标
        "points_feature": np.ndarray,  # [N, 2], intensity + elongation
    },
}
```

## 4. 标注解析逻辑

核心函数是 `waymo_decoder.decode_annos()` 和 `extract_objects()`。

`decode_annos()` 保存整帧 pose：

```python
veh_to_global = np.array(frame.pose.transform)
```

该矩阵表示当前 frame 的 vehicle -> global/world 变换。它后续主要用于多 sweep 对齐和速度坐标转换。

每个 `laser_labels` 对象解析为：

```python
{
    "id": object_id,
    "name": label.id,
    "label": label.type,
    "box": np.array([
        center_x, center_y, center_z,
        length, width, height,
        vx_ref, vy_ref,
        heading,
    ]),
    "num_points": label.num_lidar_points_in_box,
    "detection_difficulty_level": label.detection_difficulty_level,
    "combined_difficulty_level": combined_difficulty_level,
    "global_speed": [speed_x, speed_y],
    "global_accel": [accel_x, accel_y],
}
```

关键点：

- Waymo 原始 box 中心和尺寸已经在当前 vehicle 坐标系下，不需要再乘 `frame.pose`。
- `metadata.speed_x/speed_y` 是 global 坐标系速度，代码会用 `frame.pose` 的旋转矩阵逆变换到当前 vehicle/ref 坐标系。
- 速度是向量，只做旋转，不做平移。
- `num_lidar_points_in_box` 是 Waymo 数据中已有字段，不是当前代码重新统计出来的点数。

难度逻辑：

```text
num_lidar_points_in_box <= 0 -> combined_difficulty_level = 999
detection_difficulty_level == 0:
  num_lidar_points_in_box >= 5 -> Level 1
  otherwise                    -> Level 2
else:
  使用 label.detection_difficulty_level
```

## 5. infos 生成逻辑

逐帧 pickle 不是训练直接读取的索引文件。训练前还要运行：

```bash
python tools/create_data.py waymo_data_prep \
  --root_path=data/Waymo \
  --split train \
  --nsweeps=1
```

对应 `waymo_common.create_waymo_infos()`。

输出文件：

```text
infos_train_01sweeps_filter_zero_gt.pkl
infos_val_01sweeps_filter_zero_gt.pkl
infos_test_01sweeps_filter_zero_gt.pkl
infos_train_02sweeps_filter_zero_gt.pkl
...
```

每个 info 大致包含：

```python
{
    "path": "data/Waymo/{split}/lidar/seq_i_frame_j.pkl",
    "anno_path": "data/Waymo/{split}/annos/seq_i_frame_j.pkl",
    "token": "seq_i_frame_j.pkl",
    "timestamp": frame_timestamp_seconds,
    "sweeps": [...],
    "gt_boxes": np.ndarray,  # test split 没有
    "gt_names": np.ndarray,  # test split 没有
}
```

### 5.1 帧排序

`get_available_frames()` 读取 `{split}/lidar/` 下的文件名，再用 `sort_frame()` 按：

```text
seq_id * 1000 + frame_id
```

排序。因此文件名必须保持 `seq_{i}_frame_{j}.pkl` 这种格式。

### 5.2 Waymo box 到内部 box 格式

`decode_annos()` 阶段的 box 是 Waymo 原始约定：

```text
[x, y, z, length, width, height, vx, vy, heading]
```

`_fill_infos()` 会转成 CenterPoint 内部训练约定：

```text
[x, y, z, dx, dy, dz, vx, vy, yaw]
```

其中：

```text
dx = width
dy = length
dz = height
yaw = -pi / 2 - heading
```

也就是代码会交换 `length/width`：

```python
gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]
```

### 5.3 zero-point GT 过滤

非 test split 生成 infos 时会过滤：

```python
num_lidar_points_in_box > 0
```

因此 `infos_*_filter_zero_gt.pkl` 里的 `gt_boxes/gt_names` 已经不包含 0 点框。

原始 annos pickle 里仍保留所有 label，包括 `combined_difficulty_level = 999` 的对象。

## 6. 多 sweep 对齐逻辑

当 `nsweeps > 1` 时，`_fill_infos()` 会为当前参考帧向前找历史帧：

```text
seq_i_frame_j
seq_i_frame_{j-1}
seq_i_frame_{j-2}
...
```

每个历史 sweep 保存：

```python
{
    "path": curr_lidar_path,
    "transform_matrix": tm,
    "time_lag": ref_time - curr_time,
}
```

对齐矩阵为：

```text
tm = ref_from_global @ global_from_car
```

含义：

```text
历史帧 vehicle 坐标点
  -> 历史帧 vehicle 到 global
  -> global 到当前参考帧 vehicle
  -> 当前帧 vehicle 坐标点
```

如果当前帧已经是序列开头，没有足够历史帧，则用当前帧自己补齐，`transform_matrix=None`，`time_lag=0`。

## 7. Dataset 与 pipeline 加载逻辑

`WaymoDataset` 只负责加载 infos，并把样本交给 pipeline。

单 sweep 时：

```text
num_point_features = 5
points = [x, y, z, intensity, elongation]
```

多 sweep 时：

```text
num_point_features = 6
combined = [x, y, z, intensity, elongation, time_lag]
```

`LoadPointCloudFromFile` 的 Waymo 分支会：

1. 从 info 的 `path` 读取 lidar pickle。
2. 拼接 `points_xyz` 和 `points_feature`。
3. 对 intensity 做：

```python
points_feature[:, 0] = np.tanh(points_feature[:, 0])
```

4. 如果 `nsweeps > 1`，顺序读取历史 sweeps：
   - 用 `transform_matrix` 把历史点云对齐到当前帧 vehicle 坐标。
   - 给历史点追加 `time_lag`。
   - 当前帧 `time_lag = 0`。

`LoadPointCloudAnnotations` 的 Waymo 分支只从 info 中取：

```python
{
    "boxes": info["gt_boxes"],
    "names": info["gt_names"],
}
```

Waymo 的速度没有单独放到 `velocities` 字段，而是已经在 `gt_boxes[:, 6:8]` 中。

## 8. GT database

`tools/create_data.py waymo_data_prep` 在 `split == "train"` 时会额外调用：

```python
create_groundtruth_database(
    "WAYMO",
    root_path,
    infos_train_xxsweeps_filter_zero_gt.pkl,
    used_classes=["VEHICLE", "CYCLIST", "PEDESTRIAN"],
    nsweeps=nsweeps,
)
```

Waymo 目标数量很大，建库时有额外下采样策略：

- 非 4 的倍数帧会过滤掉 `VEHICLE`。
- 非 2 的倍数帧会过滤掉 `PEDESTRIAN`。
- `CYCLIST` 稀少，尽量保留。

每个 GT object 的框内点会保存成 `.bin`，且点坐标会减去 GT box 中心，变成 object-local 坐标，供后续 GT-AUG 采样插入。

## 9. 训练目标中的 box 维度

Waymo 在 `AssignLabel` 中使用 10 维训练目标：

```text
[center_offset_x, center_offset_y,
 z,
 log(dx), log(dy), log(dz),
 vx, vy,
 sin(yaw), cos(yaw)]
```

其中 `vx/vy/yaw` 来自 `_fill_infos()` 后的内部 box：

```text
[x, y, z, dx, dy, dz, vx, vy, yaw]
```

## 10. 评估输出逻辑

`WaymoDataset.evaluation()` 不直接计算 Waymo 指标，只调用：

```python
waymo_common._create_pd_detection()
```

把模型输出写成 Waymo devkit 需要的 protobuf：

```text
detection_pred.bin
tracking_pred.bin
```

模型输出的内部 box：

```text
[x, y, z, dx, dy, dz, yaw]
```

写回 Waymo 前会逆变换：

```text
heading = -yaw - pi / 2
length = dy
width = dx
height = dz
```

类别映射只覆盖检测主类：

```python
0 -> VEHICLE
1 -> PEDESTRIAN
2 -> CYCLIST
```

`SIGN` 虽然存在于 Waymo label 类型中，但当前训练/预测主路径通常不作为检测类别输出。

## 11. 易错点汇总

1. `RangeImage[..., 3]` 不是 second-return 标记，而是 NLZ 相关标记；second return 来自 `ri_return2`。
2. 当前训练点特征不保留 NLZ，也不保留 return id，最终单帧输入是 `[x, y, z, tanh(intensity), elongation]`。
3. `extract_point_cloud_from_range_image()` 当前输出是 vehicle/ego 坐标点云，不是 global 坐标。
4. TOP lidar 使用 `pixel_pose + frame_pose` 做运动补偿；非 TOP lidar 当前不传 pixel pose。
5. Waymo 原始 box 已经在 vehicle 坐标系，不需要乘 `frame.pose`；速度需要从 global 旋转到 vehicle。
6. `decode_annos()` 的 box 还是 Waymo 格式，`_fill_infos()` 后才变成 CenterPoint 内部格式。
7. `infos_*_filter_zero_gt.pkl` 会过滤 `num_lidar_points_in_box == 0` 的 GT，但原始 annos pickle 不过滤。
8. 多 sweep 对齐发生在历史帧 vehicle 坐标到当前帧 vehicle 坐标，不是把训练点云统一保存为 global 坐标。
9. Waymo 官方 mAP/mAPH 需要 Waymo devkit 计算，本仓库评估阶段主要负责写 protobuf 结果文件。
