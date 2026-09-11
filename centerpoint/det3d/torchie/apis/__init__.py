"""训练/推理高层 API 入口。

聚合分布式环境初始化、日志器、随机种子与训练流程（batch 处理、优化器构建、
train_detector）等接口，供训练脚本直接调用。推理相关接口当前被注释保留。
"""

from .env import get_root_logger, init_dist, set_random_seed
from .train import batch_processor, batch_processor_ensemble, build_optimizer, train_detector

# from .inference import init_detector, inference_detector, show_result

__all__ = [
    "init_dist",
    "get_root_logger",
    "set_random_seed",
    "train_detector",
    "build_optimizer",
    "batch_processor",
    # 'init_detector', 'inference_detector', 'show_result'
]
