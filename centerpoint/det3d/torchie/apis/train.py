"""训练流程高层 API。

提供从配置构建优化器、单 batch 前向处理、以及 train_detector 训练主函数，
把数据集、模型、优化器、学习率调度与 Trainer/Hook 组装为完整训练流程。

主要函数：
    - example_to_device / batch_processor: 把数据搬运到设备并执行前向/损失。
    - build_optimizer / build_one_cycle_optimizer: 按配置构建优化器。
    - train_detector: 训练入口，组装数据加载、模型、Hook 并启动训练。
"""

from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

try:
    import apex
except:
    print("No APEX!")

import numpy as np
import torch
from det3d.builder import _create_learning_rate_scheduler

# from det3d.datasets.kitti.eval_hooks import KittiDistEvalmAPHook, KittiEvalmAPHookV2
from det3d.core import DistOptimizerHook
from det3d.datasets import DATASETS, build_dataloader
from det3d.solver.fastai_optim import OptimWrapper
from det3d.torchie.trainer import DistSamplerSeedHook, Trainer, obj_from_dict
from det3d.utils.print_utils import metric_to_str
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .env import get_root_logger


def example_to_device(example, device=None, non_blocking=False) -> dict:
    """将单个 batch 的样本字典搬运到指定设备。

    根据字段名区分数据形态：列表型字段逐元素搬运，张量型字段整体搬运，
    calib 字段递归搬运其内部标定张量，其余字段原样保留。

    Args:
        example (dict): 单 batch 样本字典。
        device (torch.device): 目标设备（必须非空）。
        non_blocking (bool): 是否非阻塞拷贝。

    Returns:
        dict: 搬运到设备后的样本字典。
    """
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", 'points']:
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_losses(losses):
    """把模型返回的损失字典整理为总损失与标量日志变量。

    Args:
        losses (dict): 损失字典，值为张量或张量列表。

    Returns:
        tuple: (总损失 Tensor, 标量化的日志变量 OrderedDict)。
    """
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError("{} is not a tensor or list of tensors".format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    log_vars["loss"] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def parse_second_losses(losses):
    """整理 SECOND 风格模型的损失输出。

    对 losses["loss"] 求和得到总损失；其余每个键的值转换为标量列表，
    loc_loss_elem 为嵌套列表。

    Args:
        losses (dict): 模型返回的损失字典。

    Returns:
        tuple: (总损失 Tensor, 日志变量 OrderedDict)。
    """
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


def batch_processor(model, data, train_mode, **kwargs):
    """单 batch 前向处理。

    训练模式返回 (loss, log_vars, num_samples) 字典；验证模式返回模型推理结果。

    Args:
        model (nn.Module): 模型。
        data (dict): 单 batch 数据。
        train_mode (bool): 是否训练模式。
        **kwargs: 透传参数，可含 local_rank 指定设备。

    Returns:
        dict 或模型推理结果。
    """
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(data, device, non_blocking=False)

    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs
    else:
        return model(example, return_loss=False)

def batch_processor_ensemble(model1, model2, data, train_mode, **kwargs):
    """两个模型的集成推理处理函数（已废弃）。

    对两个模型预测的热图按任务取平均后交给 model1 解码为最终结果。

    注意: 该函数已废弃，入口处会直接断言失败。
    """
    assert 0, 'deprecated'
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    assert train_mode is False 

    example = example_to_device(data, device, non_blocking=False)
    del data

    preds_dicts1 = model1.pred_hm(example)
    preds_dicts2 = model2.pred_hm(example)
    
    num_task = len(preds_dicts1)

    merge_list = []

    # 对每个任务的两个模型预测取平均融合
    for task_id in range(num_task):
        preds_dict1 = preds_dicts1[task_id]
        preds_dict2 = preds_dicts2[task_id]

        for key in preds_dict1.keys():
            preds_dict1[key] = (preds_dict1[key] + preds_dict2[key]) / 2

        merge_list.append(preds_dict1)

    # 融合后的特征交给 model1 解码得到最终预测
    return model1.pred_result(example, merge_list)


def flatten_model(m):
    """递归展平模型为叶子模块列表。

    Args:
        m (nn.Module): 待展平的模型。

    Returns:
        list[nn.Module]: 模型的所有叶子模块。
    """
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def get_layer_groups(m):
    """把整个模型视为单一层组（供 fastai 优化器使用）。

    Args:
        m (nn.Module): 模型。

    Returns:
        list: 包含单个 Sequential 层组的列表。
    """
    return [nn.Sequential(*flatten_model(m))]


def build_one_cycle_optimizer(model, optimizer_config):
    """根据配置构建 one_cycle 优化器（fastai OptimWrapper 包装的 Adam）。

    Args:
        model (nn.Module): 模型。
        optimizer_config: 优化器配置（含 fixed_wd / wd / amsgrad）。

    Returns:
        OptimWrapper: 构建好的优化器包装。
    """
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_cfg.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,   # TODO: CHECKING LR HERE !!!
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=True,
    )

    return optimizer


def build_optimizer(model, optimizer_cfg):
    """根据配置构建优化器。

    Args:
        model (:obj:`nn.Module`): 需要优化参数的模型。
        optimizer_cfg (dict): 优化器配置字典。
            位置字段：
                - type: 优化器类名。
                - lr: 基础学习率。
            可选字段：
                - 对应优化器类型的任意参数，如 weight_decay、momentum 等。
                - paramwise_options: 含 3 个可选字段的字典
                  （bias_lr_mult、bias_decay_mult、norm_decay_mult）。
                  bias_lr_mult 与 bias_decay_mult 会分别乘到所有 bias 参数
                  （归一化层除外）的学习率与权重衰减上，norm_decay_mult
                  会乘到归一化层所有参数（权重与 bias）的权重衰减上。

    Returns:
        torch.optim.Optimizer: 初始化好的优化器。
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    # 未指定分参数级选项时，使用全局设置构建
    if paramwise_options is None:
        return obj_from_dict(
            optimizer_cfg, torch.optim, dict(params=model.parameters())
        )
    else:
        assert isinstance(paramwise_options, dict)
        # 取出基础学习率与权重衰减
        base_lr = optimizer_cfg["lr"]
        base_wd = optimizer_cfg.get("weight_decay", None)
        # 指定了乘子时 weight_decay 必须显式给出
        if (
            "bias_decay_mult" in paramwise_options
            or "norm_decay_mult" in paramwise_options
        ):
            assert base_wd is not None
        # 读取分参数级乘子
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        # 逐参数组设置学习率与权重衰减
        params = []
        for name, param in model.named_parameters():
            param_group = {"params": [param]}
            if not param.requires_grad:
                # FP16 训练需在 master weight 与模型权重之间拷贝梯度，
                # 保留全部参数以与 model.parameters() 逐项对齐
                params.append(param_group)
                continue

            # 归一化层覆盖其权重与 bias 的权重衰减
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # 其他层的 bias 覆盖学习率与权重衰减
            elif name.endswith(".bias"):
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult
            # 其余参数沿用全局设置

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(params, **optimizer_cfg)


def train_detector(model, dataset, cfg, distributed=False, validate=False, logger=None):
    """训练检测器的顶层入口。

    构建数据加载器与优化器/学习率调度，把模型搬到 GPU（分布式下同步 BN 并
    包装为 DDP），注册训练 Hook 后启动 Trainer 主循环。

    Args:
        model (nn.Module): 待训练模型。
        dataset (Dataset 或 list[Dataset]): 训练数据集。
        cfg: 训练配置。
        distributed (bool): 是否使用分布式训练。
        validate (bool): 是否注册验证 Hook（当前实现中被注释保留）。
        logger (logging.Logger, 可选): 外部传入日志器。
    """
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # 准备数据加载器
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=distributed
        )
        for ds in dataset
    ]

    total_steps = cfg.total_epochs * len(data_loaders[0])
    # print(f"total_steps: {total_steps}")
    if distributed:
        # 分布式下先把模型中的 BN 转换为跨进程同步的 SyncBN
        model = apex.parallel.convert_syncbn_model(model)
    if cfg.lr_config.type == "one_cycle":
        # 构建 one_cycle 优化器与配套学习率调度
        optimizer = build_one_cycle_optimizer(model, cfg.optimizer)
        lr_scheduler = _create_learning_rate_scheduler(
            optimizer, cfg.lr_config, total_steps
        )
        cfg.lr_config = None
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.drop_step, gamma=.1)
        # lr_scheduler = None
        cfg.lr_config = None 

    # 把模型放到 GPU（分布式下进一步包装为 DistributedDataParallel）
    if distributed:
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, batch_processor, optimizer, lr_scheduler, cfg.work_dir, cfg.log_level
    )

    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # 注册训练所需的默认 Hook
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if distributed:
        # 分布式下额外注册采样器种子 Hook，保证各 epoch 数据划分不同
        trainer.register_hook(DistSamplerSeedHook())

    # # register eval hooks
    # if validate:
    #     val_dataset_cfg = cfg.data.val
    #     eval_cfg = cfg.get('evaluation', {})
    #     dataset_type = DATASETS.get(val_dataset_cfg.type)
    #     trainer.register_hook(
    #         KittiEvalmAPHookV2(val_dataset_cfg, **eval_cfg))

    # 恢复或加载预训练权重
    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, local_rank=cfg.local_rank)
