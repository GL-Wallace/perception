"""训练环境初始化工具。

提供分布式进程组初始化（pytorch/mpi/slurm 启动器）、随机种子设置与根日志器
获取，通常在训练脚本最前面被调用以准备运行环境。

主要函数：
    - init_dist: 按启动器类型初始化分布式进程组。
    - set_random_seed: 统一设置各随机数生成器的种子。
    - get_root_logger: 获取根日志器（非主进程只输出 ERROR）。
"""

import logging
import os
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from det3d.torchie.trainer import get_dist_info


def init_dist(launcher, backend="nccl", **kwargs):
    """根据启动器类型初始化分布式环境。

    Args:
        launcher (str): 启动器类型，支持 pytorch / mpi / slurm。
        backend (str): 分布式后端，默认 nccl。
        **kwargs: 透传给底层 init_process_group 的额外参数。
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    if launcher == "pytorch":
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == "mpi":
        _init_dist_mpi(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError("Invalid launcher type: {}".format(launcher))


def _init_dist_pytorch(backend, **kwargs):
    """以 PyTorch 原生方式初始化分布式（读取 LOCAL_RANK 环境变量）。"""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    """MPI 启动方式（尚未实现）。"""
    raise NotImplementedError


def _init_dist_slurm(backend, port=29500, **kwargs):
    """以 Slurm 环境变量初始化分布式。

    从 SLURM_* 环境变量读取进程编号、总任务数与节点列表，并用 scontrol 解析
    主节点地址后设置 MASTER_ADDR / MASTER_PORT 等环境变量再初始化进程组。

    Args:
        backend (str): 分布式后端。
        port (int): 主节点通信端口。
    """
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    # 按进程编号取模分配到本机各 GPU
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        "scontrol show hostname {} | head -n1".format(node_list)
    )
    os.environ["MASTER_PORT"] = str(port)
    os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["RANK"] = str(proc_id)
    dist.init_process_group(backend=backend)


def set_random_seed(seed):
    """统一设置各随机数生成器种子，保证实验可复现。

    Args:
        seed (int): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    """获取根日志器。

    若根日志器尚无处理器则先进行 basicConfig；非主进程（rank != 0）仅保留
    ERROR 级别输出，避免多进程日志刷屏。

    Args:
        log_level (int): 日志级别。

    Returns:
        logging.Logger: 根日志器。
    """
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s", level=log_level
        )
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel("ERROR")
    return logger
