# CenterPoint — 基于中心点的 3D 目标检测与跟踪

本仓库是 CVPR 2021 论文 [**Center-based 3D Object Detection and Tracking**](https://arxiv.org/abs/2006.11275) 的官方实现。

CenterPoint 用「中心点」而非「轴对齐 3D box」来表示、检测和跟踪物体：先用标准 3D 主干（VoxelNet / PointPillars）从点云提取鸟瞰图（BEV）特征，再用 2D CNN 检测头预测每个类别的中心热图，并从中心位置的特征回归出 3D 尺寸、朝向与速度；两阶段版本再基于框上的点特征做精炼；跟踪则简化为贪心最近点匹配。

```bibtex
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr{\"a}henb{\"u}hl, Philipp},
  journal={CVPR},
  year={2021},
}
```

本文档是项目的**核心技术指南**，覆盖环境配置、数据准备、训练、调试优化、推理部署、代码结构、实验复现与常见问题，使读者可完全据此复现论文实验并开展二次开发。

> 论文核心指标（用于复现参考，均以 Titan RTX、batch=1 测得）：
> - Waymo 3D 检测（2 帧输入）：Veh_L2 73.0 / Ped_L2 71.5 / Cyc_L2 71.3 / **mAPH 71.9**，约 11 FPS。
> - nuScenes 3D 检测：mAP 58.0 / **NDS 65.5**。
> - nuScenes 3D 跟踪：**AMOTA 63.8**（flip test）。

---

## 目录

1. [环境配置](#1-环境配置)
2. [数据准备](#2-数据准备)
3. [训练流程](#3-训练流程)
4. [调试与优化](#4-调试与优化)
5. [推理与部署](#5-推理与部署)
6. [代码结构说明](#6-代码结构说明)
7. [实验复现](#7-实验复现)
8. [常见问题解答](#8-常见问题解答)

---

## 1. 环境配置

### 1.1 硬件要求

- **GPU**：至少 1 块支持 CUDA 的 NVIDIA GPU。官方结果在 Titan RTX（24GB）上测得；Waymo 单阶段 `samples_per_gpu=4` 需较大显存，显存不足时请降低 batch 或体素上限。
- **内存**：建议 ≥ 32GB；数据预处理（TFRecord → pickle）阶段主要吃 CPU 内存与磁盘 IO。
- **磁盘**：Waymo 完整数据集体积很大，请预留充足空间。转换后的 `lidar/` + `annos/` pickle 与 `infos_*.pkl` 同样较大。

### 1.2 软件依赖与版本

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Linux | — | 官方测试 Ubuntu 16.04 / 18.04 |
| Python | 3.6+ | 当前环境实测 3.8.20 |
| PyTorch | 1.1+ | 官方测试过 1.1 / 1.9 / 1.10.1 |
| CUDA | 10.0+ | 官方测试过 10.0 / 11.1 |
| CMake | 3.13.2+ | 编译 C++/CUDA 扩展 |
| [spconv](https://github.com/traveller59/spconv) | 1.x 或 2.x | 稀疏卷积主干 |
| [APEX](https://github.com/NVIDIA/apex) | 可选 | 仅 sync-bn；可用 torch 原生 sync-bn 替代 |
| TensorFlow | 1.15 | 仅 Waymo TFRecord 解析用 |
| [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) | 1.2.0 | Waymo 官方 devkit |

其余 Python 依赖见 [requirements.txt](requirements.txt)：包括 `numba`、`fire`、`pybind11`、`easydict`、`addict`、`nuscenes-devkit==1.0.5`、`motmetrics<=1.1.3`、`Shapely`、`pyquaternion` 等。

### 1.3 环境搭建步骤

```bash
# 1) 创建并激活 conda 环境
conda create --name centerpoint python=3.8
conda activate centerpoint

# 2) 安装与 CUDA 版本匹配的 PyTorch（示例为 CUDA 11.1）
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch

# 3) 安装 Python 依赖
pip install -r requirements.txt

# 4) Waymo 官方 devkit（仅 Waymo 数据需要）
pip install waymo-open-dataset-tf-1-15-0==1.2.0

# 5) 将项目根目录加入 PYTHONPATH（追加到 ~/.bashrc 后重新加载）
export PYTHONPATH="${PYTHONPATH}:/path/to/CenterPoint"
```

安装 `spconv`（与当前 PyTorch/CUDA 版本严格对应）：

```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *.whl
```

### 1.4 编译 CUDA 扩展

旋转 3D NMS 是**必需**的推理后处理算子；可变形卷积 `dcn` 为**可选**（仅老版本 torch 使用）：

```bash
# 旋转 NMS（必需）
cd /path/to/CenterPoint/det3d/ops/iou3d_nms
python setup.py build_ext --inplace

# 可变形卷积（可选）
cd /path/to/CenterPoint/det3d/ops/dcn
python setup.py build_ext --inplace
```

也可直接执行仓库根目录的 [setup.sh](setup.sh) 完成上述扩展编译。

### 1.5 环境验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import spconv; print('spconv ok')"
python -c "import waymo_open_dataset; print('waymo devkit ok')"
# 验证旋转 NMS CUDA 扩展可用
python -c "from det3d.ops.iou3d_nms import iou3d_nms_cuda; print('iou3d_nms ok')"
```

---

## 2. 数据准备

### 2.1 数据集组织

Waymo 原始数据为 TFRecord，需先转换为逐帧 pickle，再生成 `infos_*.pkl` 索引。期望目录结构：

```text
CenterPoint/
└── data/
    └── Waymo/            # 软链接指向真实数据根
        ├── tfrecord_training/
        ├── tfrecord_validation/
        ├── tfrecord_testing/
        ├── train/lidar/  train/annos/
        ├── val/lidar/    val/annos/
        ├── test/lidar/   test/annos/
        └── infos_{split}_{NN}sweeps_filter_zero_gt.pkl
```

### 2.2 软链接

仓库配置默认使用 `data/Waymo` 作为数据根，建议将真实数据软链接到该路径：

```bash
mkdir -p data && cd data
ln -s /actual/path/to/WAYMO_DATASET_ROOT Waymo
```

### 2.3 TFRecord → 逐帧 pickle

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

该脚本（`det3d/datasets/waymo/waymo_converter.py`）通过 `waymo_decoder` 把每帧解码为 `lidar/seq_{i}_frame_{j}.pkl` 与 `annos/seq_{i}_frame_{j}.pkl`。`CUDA_VISIBLE_DEVICES=-1` 表示纯 CPU 转换，无需 GPU。

### 2.4 生成 info 索引

```bash
# 单帧（1 sweep，检测入门推荐）
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=1
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val   --nsweeps=1
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test  --nsweeps=1

# 两帧（2 sweeps，用于速度/跟踪与两帧检测模型）
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split train --nsweeps=2
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split val   --nsweeps=2
python tools/create_data.py waymo_data_prep --root_path=data/Waymo --split test  --nsweeps=2
```

生成的 `infos_{split}_{NN}sweeps_filter_zero_gt.pkl` 记录每个样本的点云/标注路径、GT box、帧标识，以及多 sweep 场景下的前序帧路径、`transform_matrix` 与 `time_lag`。训练集同时会生成 GT database（用于 GT-AUG 目标采样增强）。

### 2.5 数据格式要点

- **点特征**：Waymo 单帧输入为 5 维 `[x, y, z, intensity, elongation]`；`intensity` 在加载时经 `tanh` 归一化。
- **box 表示**：仓库内部采用 `[x, y, z, dx, dy, dz, yaw]`；Waymo 原始标注在 info 阶段被转换为该约定（`yaw = -π/2 - heading` 并交换长宽语义）。
- **多 sweep**：历史帧点云通过 `transform_matrix` 对齐到当前帧坐标系，并追加 `time_lag` 时间差特征。

---

## 3. 训练流程

### 3.1 训练入口与命令行参数

训练入口为 [tools/train.py](tools/train.py)，参数如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `config` | 必填（位置参数） | 配置文件路径 |
| `--work_dir` | 配置内 `work_dir` | 模型与日志输出目录 |
| `--resume_from` | 无 | 断点续训的 checkpoint 路径 |
| `--validate` | False | 训练过程中是否周期性在验证集上评估 |
| `--gpus` | 1 | 非分布式训练使用的 GPU 数 |
| `--seed` | None | 随机种子，用于复现 |
| `--launcher` | `pytorch` | `pytorch` 或 `slurm` |
| `--local_rank` | 0 | 分布式场景下的本地 rank |
| `--autoscale-lr` | False | 学习率随 GPU 数线性缩放 |

### 3.2 训练命令示例

**Waymo 单阶段 VoxelNet（推荐入门，1 帧）**

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py
```

**单卡调试（smoke test）**

```bash
python tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --work_dir work_dirs/waymo_smoke --gpus 1 --seed 0
```

**断点续训**

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --resume_from work_dirs/waymo_centerpoint_voxelnet_3x/latest.pth
```

**训练周期内评估**

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py --validate
```

**用 `--autoscale-lr` 扩展 GPU 数**

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py --autoscale-lr
```

### 3.3 核心训练参数说明

以 [waymo_centerpoint_voxelnet_3x.py](configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py) 为例：

| 配置项 | 默认值 | 含义 / 可调范围 |
|---|---|---|
| `total_epochs` | 36 | 训练轮数；`3x` 即 36 epoch，`1x` 为 12 epoch |
| `samples_per_gpu` | 4 | 每卡 batch；显存不足时降为 2 或 1 |
| `workers_per_gpu` | 4 | 每卡数据加载线程数 |
| `voxel_size` | `[0.1, 0.1, 0.15]` | 体素尺寸（米） |
| `point_cloud_range` | `[-75.2, -75.2, -2, 75.2, 75.2, 4]` | 点云裁范围 |
| `optimizer.type` | `adam` | 优化器 |
| `optimizer.wd` | 0.01 | 权重衰减 |
| `lr_config.type` | `one_cycle` | OneCycle 余弦退火 |
| `lr_config.lr_max` | 0.003 | 峰值学习率（OneCycle） |
| `lr_config.div_factor` / `pct_start` | 10.0 / 0.4 | OneCycle 调度参数 |
| `grad_clip.max_norm` | 35 | 梯度裁剪 |
| `bbox_head.weight` | 2 | 回归损失权重（总损失 = hm_loss + weight · loc_loss） |
| `bbox_head.code_weights` | `[1.0]*8` | 各回归维度的加权系数 |
| `assigner.gaussian_overlap` / `min_radius` | 0.1 / 2 | 中心热图高斯半径参数 |
| `nms.nms_iou_threshold` | 0.7 | 旋转 NMS IoU 阈值 |
| `nms.score_threshold` | 0.1 | 后处理置信度阈值 |

### 3.4 训练过程监控

日志默认由 `TextLoggerHook` 输出到 `work_dir` 的日志文件，每 `log_config.interval`（默认 5）次迭代打印一次，关键指标：

- **total_loss**：总损失，训练健康的首要判断依据；
- **hm_loss**（中心热图 focal loss）：衡量中心定位分类质量；
- **loc_loss**（回归 L1）：衡量 `[offset, z, log(dim), sin/cos yaw]` 的回归质量；
- **learning rate**：随 OneCycle 先升后降。

正常训练应满足：相关损失为有限值、不出现 `NaN/Inf`、整体下降趋势。`--validate` 开启后会在验证集上补充分布式评估。

### 3.5 模型保存与 checkpoint 管理

- `checkpoint_config.interval = 1` 表示**每个 epoch 保存一次**。
- checkpoint 保存在 `work_dir` 下，形如 `latest.pth`（最新）与 `epoch_{N}.pth`（按轮次）。
- checkpoint 元信息包含配置文本与类别名（见 `train.py` 中 `checkpoint_config.meta`），可据此复现配置。
- 恢复训练用 `--resume_from` 加载 `latest.pth`；评估/导出用 `--checkpoint` 指定任意轮次权重。

---

## 4. 调试与优化

### 4.1 常见问题排查

| 症状 | 可能原因 | 处理建议 |
|---|---|---|
| `import waymo_decoder` 报 `ModuleNotFoundError` | 运行目录不对 | 从 `det3d/datasets/waymo` 目录运行转换脚本，或把该目录加入 `PYTHONPATH` |
| 转换脚本断点不命中 | `waymo_converter.py` 用多进程 `Pool` | 调试时把 `Pool.imap` 改为 `for` 循环单进程执行 |
| 训练 `loss=NaN` | 学习率过大 / 数据异常 / 坐标越界 | 降低 `lr_max`，检查点云范围与 GT 过滤，检查是否有空点云 |
| 推理 box 尺寸异常 / 朝向偏 90° | 坐标/yaw 约定错误 | 检查 `waymo_common.py` 中 `yaw = -π/2 - heading` 与长宽交换 |
| CUDA NMS 报错 | `iou3d_nms` 未编译或 CUDA/设备不匹配 | 重新 `python setup.py build_ext --inplace` 于 `det3d/ops/iou3d_nms` |
| 显存不足（OOM） | batch / 体素过大 | 降低 `samples_per_gpu` 与 `max_voxel_num`，或减小 `point_cloud_range` |

### 4.2 性能优化

- **训练效率**：多卡 `torch.distributed.launch`；配合 `--autoscale-lr` 线性扩大学习率；合理设置 `workers_per_gpu`。
- **推理效率**：用 `--speed_test` 测单卡 FPS；稀疏卷积 `spconv` 与 CUDA NMS 是关键加速点；生产部署可参考社区 TensorRT/ONNX 方案（见 §5.5）。
- **模型精度**：两阶段配置（`configs/waymo/**/two_stage/`）通过 IoU 引导置信度与 box 精炼提升精度；增大输入帧数（`nsweeps=2`）可提升 Waymo 指标；开启 flip TTA 亦有助于 nuScenes。

### 4.3 超参数调优与实验记录建议

- 优先固定数据管线与坐标约定，只调学习率、batch、epoch，避免「结构调参」。
- 记录每个实验的：配置文件路径、`work_dir`、随机种子、GPU 数、最终 loss 与验证指标、checkpoint 路径。
- 建议通过复制配置到新文件并修改，而不是直接改仓库默认配置，便于 diff 复现。
- 用 `mAP/mAPH`（Waymo）与 `NDS/mAP`（nuScenes）作为最终判据，训练期 loss 仅作健康度参考。

---

## 5. 推理与部署

### 5.1 模型加载与初始化

推理入口 [tools/dist_test.py](tools/dist_test.py) 会执行：加载配置 → 构建检测器（`build_detector`）→ `load_checkpoint` 恢复权重 → 构建验证/测试集 → 逐 batch 前向 → `dataset.evaluation` 导出结果。

命令行参数：

| 参数 | 说明 |
|---|---|
| `config` | 配置文件路径（位置参数） |
| `--work_dir` | 输出目录（必填） |
| `--checkpoint` | 权重路径，如 `work_dirs/CONFIG_NAME/latest.pth` |
| `--speed_test` | 是否速度测试（batch 恒为 1） |
| `--testset` | 是否在测试集上评估（否则验证集） |
| `--gpus` / `--launcher` / `--local_rank` | 分布式相关 |

### 5.2 批量（分布式）验证

```bash
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/dist_test.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --work_dir work_dirs/waymo_centerpoint_voxelnet_3x \
  --checkpoint work_dirs/waymo_centerpoint_voxelnet_3x/latest.pth
```

### 5.3 单卡 / 速度测试

```bash
python ./tools/dist_test.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --work_dir work_dirs/waymo_centerpoint_voxelnet_3x \
  --checkpoint work_dirs/waymo_centerpoint_voxelnet_3x/latest.pth \
  --speed_test
```

`--speed_test` 强制 batch=1 并报告单帧推断耗时，用于逐帧 FPS 对比。

### 5.4 单帧逐条离线推理与可视化

[simple_inference_waymo.py](tools/simple_inference_waymo.py) 对输入目录中每个含 `points` 键的 pickle 做体素化与检测，输出 `detections.pkl`；加 `--visual` 还会输出 `visualization.pkl`：

```bash
python tools/simple_inference_waymo.py \
  --input_data_dir data/Waymo/val/lidar \
  --output_dir work_dirs/waymo_smoke/inference \
  --config configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --checkpoint work_dirs/waymo_smoke/latest.pth \
  --visual
python tools/visual.py  # 读取 visualization.pkl 用 Open3D 展示点云与 3D 框
```

### 5.5 输出结果解析与后处理

- 检测头将热图做 `sigmoid`、尺寸做 `exp`、朝向用 `atan2(sin,cos)` 解码，并用 `out_size_factor/voxel_size/pc_range` 把特征网格还原回物理坐标，最终得到 `[x, y, z, dx, dy, dz, yaw]`。
- 后处理顺序：`score_threshold` 过滤 → `post_center_limit_range` 范围过滤 → 旋转 NMS（`rotate_nms_pcdet`，内部调 `iou3d_nms_cuda`）→ 输出最终框/分数/类别。
- 验证/测试完成后，仓库侧生成 Waymo 官方评估所需的 protobuf 二进制约为 `detection_pred.bin`（保存在 `work_dir` 下）；**官方 mAP/mAPH 需再交给 Waymo Open Dataset devkit 计算**，仓库内部 `WaymoDataset.evaluation` 只负责编码结果，不计算指标。
- 生产加速可参考社区 [CenterPointTensorRT](https://github.com/Abraham423/CenterPointTensorRT) 与 [CenterPoint-ONNX](https://github.com/CarkusL/CenterPoint)。

---

## 6. 代码结构说明

```text
CenterPoint/
├── configs/                 # 实验配置（waymo / nuscenes / mvp）
│   ├── waymo/voxelnet/      # Waymo VoxelNet 配置（含 two_stage）
│   ├── waymo/pp/            # Waymo PointPillars 配置（含 two_stage）
│   ├── nusc/                # nuScenes 配置
│   └── mvp/                 # 多模态融合(MVP)配置
├── det3d/                   # 核心库
│   ├── models/              # 模型
│   │   ├── detectors/       # VoxelNet / PointPillars / single_stage / two_stage
│   │   ├── bbox_heads/      # CenterHead（中心热图 + 属性回归）
│   │   ├── losses/          # FastFocalLoss / RegLoss
│   │   ├── backbones/       # scn（稀疏卷积）
│   │   ├── necks/           # rpn（BEV 特征金字塔）
│   │   ├── readers/         # voxel_encoder / pillar_encoder / dynamic_voxel_encoder
│   │   ├── roi_heads/       # 两阶段精炼 RoIHead
│   │   └── second_stage/    # bird_eye_view（面/点特征采样）
│   ├── datasets/            # Waymo / nuScenes 数据集与 pipeline
│   │   ├── waymo/           # waymo_converter / waymo_common / waymo_decoder / waymo
│   │   ├── nuscenes/        # nuScenes 数据集
│   │   ├── pipelines/       # loading / preprocess / formating / compose
│   │   └── loader/ utils/   # 采样器与数据工具
│   ├── core/                # 几何与体素工具
│   │   ├── utils/           # center_utils（高斯半径/热图）、circle_nms_jit 等
│   │   ├── bbox/            # box_torch_ops（旋转 NMS）、geometry、box_np_ops
│   │   ├── sampler/         # GT 数据库采样
│   │   └── input/           # voxel_generator
│   ├── ops/                 # CUDA 算子封装（iou3d_nms / dcn / point_cloud）
│   ├── solver/              # 优化器与学习率调度（fastai 风格 OneCycle）
│   ├── torchie/             # 训练框架基础设施（Trainer / Hook / Config / fileio / parallel）
│   └── utils/               # 分布式、checkpoint、config_tool、buildtools 等
├── tools/                   # 训练 / 测试 / 数据生成 / 可视化 / 跟踪脚本
│   ├── train.py             # 训练入口
│   ├── dist_test.py         # 分布式验证/测试入口
│   ├── create_data.py       # 数据预处理入口（fire 子命令）
│   ├── simple_inference_waymo.py  # Waymo 逐帧离线推理
│   ├── visual.py / demo.py  # 可视化 / demo
│   ├── waymo_tracking/      # Waymo 跟踪
│   └── nusc_tracking/       # nuScenes 跟踪
├── docs/                    # 安装 / 数据 / 注释规范等文档
├── data/                    # 数据软链接（data/Waymo）
└── work_dirs/               # 训练产物（模型与日志）
```

**关键数据/调用链**：

```text
Waymo TFRecord
→ waymo_converter.py（逐帧 pickle）
→ create_data.py → infos_*.pkl
→ WaymoDataset + pipelines（loading → preprocess → voxelization → AssignLabel → reformat）
→ reader → backbone(scn) → neck(rpn) → CenterHead
→ loss(FastFocalLoss + RegLoss)   # 训练
→ CenterHead.predict → 解码 → rotate_nms_pcdet   # 推理
→ detection_pred.bin → Waymo devkit 计算指标
```

**核心函数速查**：

- [CenterHead](det3d/models/bbox_heads/center_head.py)：`forward` / `loss` / `predict` / `post_processing`。
- [AssignLabel](det3d/datasets/pipelines/preprocess.py)：由 GT box 生成 `hm/anno_box/ind/mask/cat`。
- [center_utils](det3d/core/utils/center_utils.py)：`gaussian_radius`、`draw_umich_gaussian`、`_transpose_and_gather_feat`。
- [centernet_loss](det3d/models/losses/centernet_loss.py)：`FastFocalLoss`、`RegLoss`。
- [box_torch_ops](det3d/core/bbox/box_torch_ops.py)：`rotate_nms_pcdet`。

> 完整的中文逐模块注释已覆盖全部 207 个源码文件，注释规范见 [docs/注释规范.md](docs/注释规范.md)。

---

## 7. 实验复现

### 7.1 Waymo 单阶段（1 帧，VoxelNet）

```bash
# 训练（4 卡，36 epoch）
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py

# 验证（4 卡）
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/dist_test.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py \
  --work_dir work_dirs/waymo_centerpoint_voxelnet_3x \
  --checkpoint work_dirs/waymo_centerpoint_voxelnet_3x/latest.pth
```

### 7.2 Waymo 两阶段（最终模型）

两阶段是「先训单阶段 → 再训精炼头」的两步流程：

```bash
# 第一步：训练单阶段
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/waymo_centerpoint_voxelnet_3x.py

# 第二步：基于单阶段权重训练两阶段精炼头
python -m torch.distributed.launch --nproc_per_node=4 \
  ./tools/train.py configs/waymo/voxelnet/two_stage/waymo_centerpoint_voxelnet_two_stage_bev_5point_ft_6epoch_freeze.py
```

### 7.3 生成 Waymo 评估 GT 与提交文件

```bash
# 生成本地 GT protobuf（供 devkit 本地评估）
python det3d/datasets/waymo/waymo_common.py \
  --info_path data/Waymo/infos_val_01sweeps_filter_zero_gt.pkl \
  --result_path data/Waymo/ --gt
```

验证集推理会在 `work_dir` 生成 Waymo 格式的 `detection_pred.bin`，测试集推理附加 `--testset`。官方 mAP/mAPH 需用 Waymo Open Dataset devkit 计算（参见其 [quick_start](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md)）。

### 7.4 nuScenes 与跟踪

nuScenes 数据准备、训练与跟踪命令见 [configs/nusc/README.md](configs/nusc/README.md) 与 `docs/NUSC.md`；Waymo 跟踪入口见 [tools/waymo_tracking/test.py](tools/waymo_tracking/test.py)。跟踪依赖 [dist_test.py](tools/dist_test.py) 生成的中间预测文件。

### 7.5 复现预期指标

在 Titan RTX、batch=1 下，复现目标应接近：Waymo 单帧 mAPH 69.0、两帧 mAPH 71.9；nuScenes NDS 65.5、跟踪 AMOTA 63.8。实际结果会受数据版本、随机种子与硬件事先影响，可作为校准参考而非强制断崖。

---

## 8. 常见问题解答

**Q1：`data/Waymo` 与 `dataset/waymo` 是什么关系？**
`data/Waymo` 是软链接，指向 `dataset/waymo`（或你的真实数据根）。两者是同一份物理数据，保留软链接可让配置中的相对路径 `data/Waymo` 无需修改。

**Q2：`my_preds.bin` 还是 `detection_pred.bin`？**
`docs/WAYMO.md` 旧文本写的是 `my_preds.bin`，但当前源码 `_create_pd_detection` 实际输出 `detection_pred.bin`，请以源码为准。

**Q3：为什么 Waymo 点特征是 5 维？**
单帧 Waymo 为 `[x, y, z, intensity, elongation]`；多 sweep 还会追加 `time_lag`，因此两帧模型输入维度更多。

**Q4：如何单步调试数据转换脚本？**
`waymo_converter.py` 内部用多进程 `Pool`，断点不会命中子进程。调试时把 `Pool.imap` 改为 `for i in range(len(fnames)): convert(i)` 的单进程循环，再设断点。

**Q5：训练画出 `NaN` 怎么办？**
先确认点云范围与 GT 过滤正常、空点云被剔除、学习率不过大；逐步二分定位到具体 batch 与模块。

**Q6：推理框朝向整体偏 90°？**
检查 `waymo_common.py` 中 `yaw = -π/2 - heading` 与长宽交换是否正确执行，以及可视化脚本的坐标约定是否一致。

**Q7：`--testset` 与 `--speed_test` 区别？**
`--testset` 切到测试集（无 GT）；`--speed_test` 强制 batch=1 并统计单帧耗时，二者可独立使用。

**Q8：官方指标在哪算？**
仓库只负责把模型输出编码为 Waymo/nuScenes 要求的格式；Waymo 官方 mAP/mAPH 与 nuScenes 官方 NDS 需分别用各自官方 devkit 计算。

---

## 许可与致谢

本项目以 MIT License 发布（见 [LICENSE](LICENSE)），基于 [det3d](https://github.com/poodarchu/det3d) 并融合了 [CenterNet](https://github.com/xingyizhou/CenterNet) 与 [CenterTrack](https://github.com/xingyizhou/CenterTrack) 的代码。注意 nuScenes 与 Waymo 数据集均遵循非商业许可。