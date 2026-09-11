"""训练检查点的加载与保存工具。

提供 state_dict 的灵活加载（含 spconv 权重形状适配）、从本地文件 / URL /
模型库加载 checkpoint、以及把模型与优化器 state_dict 保存为单文件 checkpoint。

主要函数：
    - load_checkpoint / save_checkpoint: 加载与保存 checkpoint。
    - load_state_dict: 把 state_dict 载入 module，容忍缺失/多余键并输出差异。
    - find_all_spconv_keys: 收集需要转置的 spconv 卷积权重键。
    - weights_to_cpu: 将 state_dict 搬运到 CPU。
    - load_url_dist: 分布式场景下仅在本地 rank 0 下载一次权重。

依赖 det3d.torchie 提供目录工具，依赖 terminaltables 格式化形状不匹配表格。
"""

import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module

import torch
import torchvision
from det3d import torchie
from terminaltables import AsciiTable
from torch.utils import model_zoo

from .utils import get_dist_info

open_mmlab_model_urls = {
    "vgg16_caffe": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth",  # noqa: E501
    "resnet50_caffe": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_caffe-788b5fa3.pth",  # noqa: E501
    "resnet101_caffe": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth",  # noqa: E501
    "resnext50_32x4d": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth",  # noqa: E501
    "resnext101_32x4d": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth",  # noqa: E501
    "resnext101_64x4d": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth",  # noqa: E501
    "contrib/resnet50_gn": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_thangvubk-ad1730dd.pth",  # noqa: E501
    "detectron/resnet50_gn": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn-9186a21c.pth",  # noqa: E501
    "detectron/resnet101_gn": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn-cac0ab98.pth",  # noqa: E501
    "jhu/resnet50_gn_ws": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth",  # noqa: E501
    "jhu/resnet101_gn_ws": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn_ws-3e3c308c.pth",  # noqa: E501
    "jhu/resnext50_32x4d_gn_ws": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth",  # noqa: E501
    "jhu/resnext101_32x4d_gn_ws": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth",  # noqa: E501
    "jhu/resnext50_32x4d_gn": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn-c7e8b754.pth",  # noqa: E501
    "jhu/resnext101_32x4d_gn": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn-ac3bb84e.pth",  # noqa: E501
    "msra/hrnetv2_w18": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w18-00eb2006.pth",  # noqa: E501
    "msra/hrnetv2_w32": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w32-dc9eeb4f.pth",  # noqa: E501
    "msra/hrnetv2_w40": "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/hrnetv2_w40-ed0b031c.pth",  # noqa: E501
    "bninception_caffe": "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/bn_inception_caffe-ed2e8665.pth",  # noqa: E501
    "kin400/i3d_r50_f32s2_k400": "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/i3d_r50_f32s2_k400-2c57e077.pth",  # noqa: E501
    "kin400/nl3d_r50_f32s2_k400": "https://open-mmlab.s3.ap-northeast-2.amazonaws.com/pretrain/third_party/nl3d_r50_f32s2_k400-fa7e7caa.pth",  # noqa: E501
}  # yapf: disable

import torch.nn as nn 
from typing import Set

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """递归收集模型中所有 spconv 卷积层的权重键。

    用于在加载旧版 spconv 权重时定位需要做通道重排的卷积权重名。

    Args:
        model (nn.Module): 待扫描的模型。
        prefix (str): 当前递归层级的前缀。

    Returns:
        Set[str]: 所有 spconv 卷积权重对应的 state_dict 键名集合。
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys


def load_state_dict(module, state_dict, strict=False, logger=None):
    """将 state_dict 加载到模块中。

    相比 strict 加载，本函数会收集缺失/多余键与形状不匹配项并输出提示；
    对 spconv 卷积权重还会在加载前尝试不同通道重排方式以适配版本差异。

    Args:
        module (nn.Module): 目标模块。
        state_dict (dict): 待加载的权重字典。
        strict (bool): 是否存在不匹配时需要抛出异常。
        logger (logging.Logger, 可选): 不匹配时接收警告的日志器。
    """
    unexpected_keys = []
    shape_mismatch_pairs = []

    own_state = module.state_dict()

    spconv_keys = find_all_spconv_keys(module)

    for name, param in state_dict.items():

        if name in spconv_keys and name in own_state and own_state[name].shape != param.shape:
            # 不同 spconv 版本卷积权重的通道顺序不同：
            # 若来自 spconv 1.x，则按 (k1,k2,k3,c_in,c_out) 转置为
            # (k1,k2,k3,c_out,c_in)，或按 implicit 方式重排为 (c_out,k1,k2,k3,c_in)。

            param_native = param.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) -> (k1, k2, k3, c_out, c_in)
            if param_native.shape == own_state[name].shape:
                param = param_native.contiguous()
            else:
                assert param.shape.__len__() == 5, 'currently only spconv 3D is supported'
                param_implicit = param.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) -> (c_out, k1, k2, k3, c_in)
                if param_implicit.shape == own_state[name].shape:
                    param = param_implicit.contiguous()


        # 兼容旧 voxelnet 权重：目标 module 不存在的键直接跳过并记录
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # 兼容以 Parameter 形式序列化的权重，取其底层 data
            param = param.data
        if param.size() != own_state[name].size():
            shape_mismatch_pairs.append([name, own_state[name].size(), param.size()])
            continue
        own_state[name].copy_(param)

    all_missing_keys = set(own_state.keys()) - set(state_dict.keys())
    # 忽略 BatchNorm 的 num_batches_tracked 计数缓冲
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    err_msg = []
    if unexpected_keys:
        err_msg.append(
            "unexpected key in source state_dict: {}\n".format(
                ", ".join(unexpected_keys)
            )
        )
    if missing_keys:
        err_msg.append(
            "missing keys in source state_dict: {}\n".format(", ".join(missing_keys))
        )
    if shape_mismatch_pairs:
        mismatch_info = "these keys have mismatched shape:\n"
        header = ["key", "expected shape", "loaded shape"]
        table_data = [header] + shape_mismatch_pairs
        table = AsciiTable(table_data)
        err_msg.append(mismatch_info + table.table)

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_url_dist(url):
    """分布式场景下下载 checkpoint，仅本地 rank 0 真正下载。

    Args:
        url (str): 权重文件 URL。

    Returns:
        checkpoint: 下载得到的 checkpoint 对象。
    """
    rank, world_size = get_dist_info()
    rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url)
    if world_size > 1:
        # 非主 rank 等待主 rank 下载完成后再各自加载（避免重复下载）
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url)
    return checkpoint


def get_torchvision_models():
    """收集 torchvision 中各子模块提供的预训练权重 URL。

    Returns:
        dict: 模型名到权重 URL 的映射。
    """
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module("torchvision.models.{}".format(name))
        if hasattr(_zoo, "model_urls"):
            _urls = getattr(_zoo, "model_urls")
            model_urls.update(_urls)
    return model_urls


def load_checkpoint(model, filename, map_location='cpu', strict=False, logger=None):
    """从文件或 URL 加载 checkpoint。

    支持 modelzoo://、torchvision://、open-mmlab://、http(s):// 及本地文件路径；
    随后剥离可能的 "module." 前缀并把 state_dict 载入模型。

    Args:
        model (Module): 需要加载权重的模块。
        filename (str): 文件路径或 URL 或 modelzoo://xxxxxxx 形式。
        map_location (str): 与 :func:`torch.load` 相同的设备映射参数。
        strict (bool): 是否允许模型与 checkpoint 参数不一致。
        logger (:mod:`logging.Logger` 或 None): 接收错误信息的日志器。

    Returns:
        dict 或 OrderedDict: 加载得到的 checkpoint。
    """
    # 从 modelzoo / 远程 URL / 本地文件加载 checkpoint
    if filename.startswith("modelzoo://"):
        warnings.warn(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead'
        )
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith("torchvision://"):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
    elif filename.startswith("open-mmlab://"):
        model_name = filename[13:]
        checkpoint = load_url_dist(open_mmlab_model_urls[model_name])
    elif filename.startswith(("http://", "https://")):
        checkpoint = load_url_dist(filename)
    else:
        if not osp.isfile(filename):
            raise IOError("{} is not a checkpoint file".format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # 从 checkpoint 中取出 state_dict
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))
    # 若键带 "module." 前缀（DataParallel/DDP 包装导致），统一剥离
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
    # 载入 state_dict（DDP 包装下取被包装的 module）
    if hasattr(model, "module"):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """把模型 state_dict 中的张量搬运到 CPU。

    Args:
        state_dict (OrderedDict): 位于 GPU 上的模型权重。

    Returns:
        OrderedDict: 搬运到 CPU 后的模型权重。
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """把 checkpoint 保存到文件。

    checkpoint 包含三个字段：``meta``、``state_dict`` 与 ``optimizer``；
    默认 meta 会附带版本与时间信息。

    Args:
        model (Module): 待保存参数的模块。
        filename (str): checkpoint 文件名。
        optimizer (:obj:`Optimizer`, 可选): 需要一并保存的优化器。
        meta (dict, 可选): 需要保存在 checkpoint 中的元信息。
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None, but got {}".format(type(meta)))

    torchie.mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, "module"):
        model = model.module

    checkpoint = {"meta": meta, "state_dict": weights_to_cpu(model.state_dict())}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, filename)
