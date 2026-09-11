"""多 GPU 并行测试工具。

通过多进程（spawn 上下文）把不同样本分发给不同 GPU 上的 worker 进程执行前向
推理，用队列按样本索引回传结果，适用于数据集较大、需多卡加速的离线评测。

主要函数：
    - worker_func: 单个 worker 进程的循环，加载模型并逐样本推理。
    - parallel_test: 启动多个 worker 进程并分发样本。

注意：本文件的 parallel_test 目前只完成了 worker 启动与样本入队，缺少结果
汇总与返回逻辑，调用方式尚不完整。
"""

import multiprocessing

import torch
from det3d import torchie

from .checkpoint import load_checkpoint


def worker_func(
    model_cls,
    model_kwargs,
    checkpoint,
    dataset,
    data_func,
    gpu_id,
    idx_queue,
    result_queue,
):
    """单个并行测试 worker 的执行体。

    初始化模型并加载 checkpoint 到指定 GPU，循环从 idx_queue 取样本索引，
    推理后把 (idx, result) 放入 result_queue。

    Args:
        model_cls (type): 模型类。
        model_kwargs (dict): 实例化模型的参数。
        checkpoint (str): checkpoint 文件路径。
        dataset (Dataset): 待测试的数据集。
        data_func (callable): 生成模型输入的预处理函数。
        gpu_id (int): 本 worker 绑定的 GPU 编号。
        idx_queue (Queue): 待处理的样本索引队列。
        result_queue (Queue): 返回结果的队列。
    """
    model = model_cls(**model_kwargs)
    load_checkpoint(model, checkpoint, map_location="cpu")
    torch.cuda.set_device(gpu_id)
    model.cuda()
    model.eval()
    with torch.no_grad():
        while True:
            idx = idx_queue.get()
            data = dataset[idx]
            result = model(**data_func(data, gpu_id))
            result_queue.put((idx, result))


def parallel_test(
    model_cls, model_kwargs, checkpoint, dataset, data_func, gpus, workers_per_gpu=1
):
    """在多个 GPU 上并行测试。

    Args:
        model_cls (type): 模型类。
        model_kwargs (dict): 实例化模型的参数。
        checkpoint (str): checkpoint 文件路径。
        dataset (:obj:`Dataset`): 待测试的数据集。
        data_func (callable): 生成模型输入的函数。
        gpus (list[int]): 需要使用的 GPU 编号列表。
        workers_per_gpu (int): 每张 GPU 上的进程数，允许每卡运行多个 worker。

    Returns:
        list: 测试结果。
    """
    ctx = multiprocessing.get_context("spawn")
    idx_queue = ctx.Queue()
    result_queue = ctx.Queue()
    num_workers = len(gpus) * workers_per_gpu
    workers = [
        ctx.Process(
            target=worker_func,
            args=(
                model_cls,
                model_kwargs,
                checkpoint,
                dataset,
                data_func,
                # 轮转分配 GPU：第 i 个 worker 使用 gpus[i % len(gpus)]
                gpus[i % len(gpus)],
                idx_queue,
                result_queue,
            ),
        )
        for i in range(num_workers)
    ]
    for w in workers:
        # 设为 daemon 以便主进程退出时自动回收
        w.daemon = True
        w.start()

    # 把所有样本索引依次放入队列，供各 worker 抢取
    for i in range(len(dataset)):
        idx_queue.put(i)

    results = [None for _ in range(len(dataset))]
