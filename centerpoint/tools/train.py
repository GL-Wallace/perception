import argparse
import json
import os
import sys

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
import torch.distributed as dist
import subprocess

def parse_args():
    """解析训练脚本的命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象，包含配置文件路径、工作目录、
            断点续训路径、GPU 数量、随机种子、启动方式等。

    注意:
        若环境变量中不存在 LOCAL_RANK，则用传入的 --local_rank 补齐，
        保证后续分布式初始化能读取到一致的本地 rank。
    """
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    """训练主流程。

    加载配置、初始化分布式环境、设置随机种子、构建模型与数据集，最后调用
    train_detector 启动训练。具体步骤为：

    1. 解析命令行参数并用其对配置进行覆盖（work_dir、resume_from）；
    2. 根据环境变量判断是否分布式训练，并完成对应启动方式（pytorch/slurm）的初始化；
    3. 可选地随 GPU 数量缩放学习率；
    4. 构建检测器与数据集，写入 checkpoint 元信息；
    5. 调用 train_detector 开始训练。
    """
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)

    # 用命令行参数覆盖配置文件中的对应字段
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    distributed = False
    if "WORLD_SIZE" in os.environ:
        # 由启动方式（如 torch.distributed.launch）注入 WORLD_SIZE 判断是否多卡
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            cfg.local_rank = args.local_rank
        elif args.launcher == "slurm":
            # Slurm 环境下从环境变量读取进程编号、任务总数与节点列表
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            cfg.gpus = num_gpus
            # 每个进程绑定到其进程号对 GPU 数取模所得的设备
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # 指定 master 节点的通信端口
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # 直接沿用环境变量中已有的 MASTER_PORT
            else:
                # 29500 是 torch.distributed 默认端口，这里改为 29501 避免冲突
                os.environ["MASTER_PORT"] = "29501"
            # 若环境变量中已有 MASTER_ADDR 则直接沿用，否则取首节点的 hostname
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            # 补齐分布式初始化所需的环境变量
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            cfg.local_rank = int(os.environ["LOCAL_RANK"])

        # 分布式场景下以进程组规模作为使用的 GPU 数
        cfg.gpus = dist.get_world_size()
    else:
        cfg.local_rank = args.local_rank 

    if args.autoscale_lr:
        # 学习率随 GPU 数量线性缩放，保持等效 batch 的学习率一致
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # 在其他步骤之前初始化日志器
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # 主进程备份关键文件（此处备份拷贝逻辑已被注释掉，仅创建目录）
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # 设置随机种子以保证结果可复现
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    # workflow 长度为 2 表示训练过程中需要验证，此时额外构建验证集
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # 将配置文件内容与类别名写入 checkpoint 元信息，便于复现与可视化
        cfg.checkpoint_config.meta = dict(
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )

    # 为便于可视化，给模型附加类别名属性
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


if __name__ == "__main__":
    main()
