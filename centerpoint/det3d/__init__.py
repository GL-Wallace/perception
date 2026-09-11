"""det3d 核心库包。

CenterPoint 的核心实现所在包，按职责划分为：
    - models: 模型组装、检测头、损失、reader/backbone/neck、两阶段精炼。
    - datasets: Waymo/nuScenes 数据集与数据 pipeline。
    - core: 体素生成、旋转框几何、目标采样工具。
    - ops: CUDA 算子（旋转框 NMS、可变形卷积、点云算子）封装。
    - solver: 优化器与学习率调度。
    - torchie: 训练框架基础设施（Trainer/Hook/配置/文件 IO 等）。
    - utils: 分布式、checkpoint、配置等通用工具。
"""