"""训练框架多 GPU 通信与分布式训练的基础工具集。

提供分布式环境查询（rank/world_size）、主进程装饰器、配置转对象，以及
all_gather / reduce 等跨进程通信原语，供 Trainer 与 Hook 在分布式训练中使用。

主要函数：
    - get_host_info / get_dist_info: 查询主机信息与分布式 rank/world_size。
    - master_only: 仅允许主进程（rank 0）执行被装饰函数。
    - obj_from_dict: 根据含 type 键的字典实例化对象。
    - synchronize / all_gather / reduce_dict: 跨进程同步、收集与规约。
"""

import functools
import pickle
import sys
import time
from getpass import getuser
from socket import gethostname

import torch
import torch.distributed as dist
from det3d import torchie


def get_host_info():
    """获取当前主机信息。

    Returns:
        str: 形如 "用户名@主机名" 的字符串，用于日志标识运行节点。
    """
    return "{}@{}".format(getuser(), gethostname())


def get_dist_info():
    """获取当前进程的分布式信息。

    Returns:
        tuple: (rank, world_size)。未初始化分布式时返回 (0, 1)。
    """
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    """装饰器：确保函数仅在主进程（rank 0）执行。

    依赖 :func:`get_dist_info` 判断当前进程 rank，非主进程直接跳过执行。

    Args:
        func (callable): 被装饰函数。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def get_time_str():
    """获取当前本地时间的字符串表示。

    Returns:
        str: 形如 "%Y%m%d_%H%M%S" 的时间戳字符串。
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def obj_from_dict(info, parent=None, default_args=None):
    """根据字典实例化对象。

    字典必须包含 "type" 键以指明对象类型：
    - 若 type 为字符串且 parent 非空，则从 parent 中按属性名取类；
    - 若 type 为字符串且 parent 为空，则从已导入模块 sys.modules 中取模块对象；
    - 若 type 为类，则直接使用。

    Args:
        info (dict): 对象类型与构造参数。
        parent (:class:`module`): 用于查找 name 的模块对象。
        default_args (dict, 可选): 默认参数，若 info 中未指定则补齐。
    """
    assert isinstance(info, dict) and "type" in info
    assert isinstance(default_args, dict) or default_args is None
    args = info.copy()
    obj_type = args.pop("type")
    if torchie.is_str(obj_type):
        if parent is not None:
            obj_type = getattr(parent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError(
            "type must be a str or valid type, but got {}".format(type(obj_type))
        )
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def get_world_size():
    """获取分布式进程总数。

    Returns:
        int: 进程总数；分布式不可用或未初始化时为 1。
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程的 rank。

    Returns:
        int: 当前进程 rank；分布式不可用或未初始化时为 0。
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """判断当前进程是否为主进程。

    Returns:
        bool: rank 是否为 0。
    """
    return get_rank() == 0


def synchronize():
    """在分布式训练中同步（barrier）所有进程。

    仅当分布式已初始化且进程数大于 1 时才执行 barrier。
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """对任意可 pickle 化的对象执行 all_gather。

    先把对象序列化为字节张量，再借助 PyTorch 的 all_gather 收集各 rank 数据。
    由于 all_gather 不支持不同长度的张量，因此先统一到全局最大长度（尾部补零），
    收到后再按各自原始长度截断并反序列化。

    Args:
        data: 任意可 pickle 化的对象。

    Returns:
        list[data]: 每个 rank 收集到的数据组成的列表。
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 序列化为 ByteTensor，以便后续 all_gather
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # 先收集各 rank 序列化后的长度
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 由于 all_gather 要求各张量等长，短张量需在尾部补零对齐
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        # 按真实长度截断，丢弃补零部分后反序列化
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """把各进程的字典值规约到主进程。

    所有值按 rank 加到 rank 0；若 average 为 True 再除以 world_size 求均值。

    Args:
        input_dict (dict): 每个值都会被规约的字典。
        average (bool): 是否求平均（False 则仅求和）。

    Returns:
        dict: 与新字段一致的规约后字典。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # 对键排序，保证各进程处理顺序一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # 求和只在主进程累积，因此仅主进程需要除以 world_size
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
