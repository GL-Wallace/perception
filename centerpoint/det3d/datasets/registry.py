"""数据集与 pipeline 算子的注册表。

DATASETS 用于注册/查找数据集类，PIPELINES 用于注册/查找数据预处理各阶段算子，
二者都由 det3d.utils.registry.Registry 管理，供 build_from_cfg 按配置 name 实例化。
"""
from det3d.utils.registry import Registry

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")
