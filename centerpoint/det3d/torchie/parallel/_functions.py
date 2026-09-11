"""自定义 Scatter 实现。

支持把单个张量（或张量列表）散列到多个 GPU，并在 CPU->GPU 拷贝时使用每个
目标设备的后台流，最后与主 stream 同步，减少跨设备拷贝开销。

主要函数/类：
    - scatter: 递归散列张量/列表到各设备。
    - synchronize_stream: 让各设备主 stream 等待拷贝流。
    - get_input_device: 判断输入的原始设备。
    - Scatter: 供前向 autograd 调用 Scatter.forward 的静态入口。
"""

import torch
from torch.nn.parallel._functions import _get_stream


def scatter(input, devices, streams=None):
    """把张量（或张量列表）散列到多个 GPU。

    Args:
        input (Tensor 或 list[Tensor]): 待散列的输入。
        devices (list[int]): 目标设备列表。
        streams: 每个设备对应的 CUDA 流（可选）。
    """
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        # 列表按 chunk 均匀切分到各设备后递归处理
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            scatter(input[i], [devices[i // chunk_size]], [streams[i // chunk_size]])
            for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # 空张量无需流拷贝
        stream = streams[0] if output.numel() > 0 else None
        with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
            output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise Exception("Unknown type {}.".format(type(input)))


def synchronize_stream(output, devices, streams):
    """让各目标设备的主 stream 等待对应的拷贝 stream 完成。"""
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(
                    output[i * chunk_size + j], [devices[i]], [streams[i]]
                )
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception("Unknown type {}.".format(type(output)))


def get_input_device(input):
    """获取输入张量所在的设备编号（CPU 返回 -1）。"""
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception("Unknown type {}.".format(type(input)))


class Scatter(object):
    """散列输入的静态入口。

    forward 判断输入是否在 CPU，若是则为每个目标设备申请后台流，
    完成后与对应主 stream 同步；返回各设备结果的元组。
    """

    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # 输入在 CPU：为每个目标设备分配后台流执行拷贝
            streams = [_get_stream(device) for device in target_gpus]

        outputs = scatter(input, target_gpus, streams)
        # 等待拷贝流与各设备主 stream 同步
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)
