# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""checkpoint 保存/加载与训练日志工具。

提供 state_dict 的前缀对齐/按后缀匹配加载、Checkpointer/det3dCheckpointer
检查点管理器，以及基于 SummaryWriter 的 Writer 日志记录器。

主要函数/类：
    - flat_nested_json_dict / metric_to_str: 指标展平与格式化。
    - align_and_update_state_dicts: 按后缀匹配对齐模型与加载的权重。
    - load_state_dict / finetune_load_state_dict: 权重加载。
    - Checkpointer / det3dCheckpointer: 检查点保存与加载。
    - Writer: tensorboard 与文本日志记录。
"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import torch
from tensorboardX import SummaryWriter


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    """递归展平嵌套 dict（内部使用）。"""
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """把嵌套 json 风格 dict 展平为单层 dict（浅拷贝）。"""
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, str(k))
        else:
            flatted[str(k)] = v
    return flatted


def metric_to_str(metrics, sep="."):
    """把指标 dict 格式化为逗号分隔的字符串。

    浮点数保留 4 位有效数字；浮点列表/元组以 [...] 形式输出。
    """
    flatted_metrics = flat_nested_json_dict(metrics, sep)
    metrics_str_list = []
    for k, v in flatted_metrics.items():
        if isinstance(v, float):
            metrics_str_list.append(f"{k}={v:.4}")
        elif isinstance(v, (list, tuple)):
            if v and isinstance(v[0], float):
                v_str = ", ".join([f"{e:.4}" for e in v])
                metrics_str_list.append(f"{k}=[{v_str}]")
            else:
                metrics_str_list.append(f"{k}={v}")
        else:
            metrics_str_list.append(f"{k}={v}")
    return ", ".join(metrics_str_list)


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=None):
    """把加载的权重按键名后缀对齐到模型权重。

    策略：模型键名可能带有额外前缀（如 backbone[0].body.res2.conv1.weight），
    而预训练权重只含 res2.conv1.weight。对每个模型权重，在所有加载键中寻找其
    后缀匹配项；若有多个匹配，取对应名称最长者。匹配不到的键保持原值不变。
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # 构造匹配矩阵：entry (i, j) 为能匹配的 loaded_key 字符串长度。
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # 无匹配的条目索引置 -1。
    idxs[max_match_size == 0] = -1

    # 仅用于日志对齐。
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    if logger is None:
        logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    """若所有键都以给定前缀开头，则去除该前缀后返回新的 OrderedDict。"""
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, logger=None):
    """加载权重到模型（严格匹配）。

    若权重来自被 DataParallel/DistributedDataParallel 包裹的模型，先去除
    "module." 前缀，再按后缀对齐加载。
    """
    model_state_dict = model.state_dict()
    # 权重若来自 DataParallel / DistributedDataParallel 序列化，
    # 先去 "module." 前缀再做匹配。
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=logger)

    # 严格加载
    model.load_state_dict(model_state_dict)


def finetune_load_state_dict(model, loaded_state_dict, logger=None):
    """微调加载权重：在严格加载前过滤掉以 rpn.tasks 开头的键。"""
    model_state_dict = model.state_dict()
    # 权重若来自 DataParallel / DistributedDataParallel 序列化，
    # 先去 "module." 前缀再做匹配。
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    loaded_state_dict = {
        k: v for k, v in loaded_state_dict.items() if not k.startswith("rpn.tasks")
    }
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, logger=logger)

    # 严格加载
    model.load_state_dict(model_state_dict)


class Checkpointer(object):
    """检查点管理器：负责模型的保存、加载与 last_checkpoint 标记。"""

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        ckpt_path=None,
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pretrained_path = ckpt_path  # 是否为预训练权重
        self.finetune = False
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        """保存模型（及可选的优化器/调度器）状态到 save_dir。"""
        self.logger.info(name)
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            print(dir(self.scheduler))
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        """加载检查点：恢复模型，并恢复存在的优化器与调度器状态。"""
        if f is not None:
            f = self.get_checkpoint_file(f)
        elif self.has_checkpoint(self.save_dir):
            # 用已有检查点覆盖参数。
            f = self.get_checkpoint_file(self.save_dir)

        if not f:
            # 未找到检查点，从头初始化。
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # 返回剩余检查点数据。
        return checkpoint

    def finetune_load(self, ckpt_path=None, f=None):
        """加载预训练权重用于微调。"""
        if ckpt_path is not None:
            self.pretrained_path = ckpt_path
            self.finetune = True
            f = self.get_checkpoint_file(ckpt_path)
        assert f is not None, "Finetune should provide a valid ckpt path"
        self.logger.info("Loading pretrained model from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def has_checkpoint(self, save_dir):
        """判断 save_dir 下是否存在 last_checkpoint 标记文件。"""
        save_file = os.path.join(save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self, save_dir):
        """读取 last_checkpoint 中记录的最新检查点文件名。"""
        save_file = os.path.join(save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # 文件不存在（可能被其他进程删除）。
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        """把最新检查点文件名写入 last_checkpoint。"""
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        if self.finetune:
            finetune_load_state_dict(
                self.model, checkpoint.pop("model"), logger=self.logger
            )
        else:
            load_state_dict(self.model, checkpoint.pop("model"), logger=self.logger)


class det3dCheckpointer(Checkpointer):
    """CenterPoint 专用检查点管理器：兼容无 "model" 键的旧检查点格式。"""

    def __init__(
        self,
        # cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(det3dCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        # self.cfg = cfg.clone()
        # self.writer = Writer(save_dir)
        self.logger = logger

    def _load_file(self, f):
        # 加载原生 detectron.pytorch 检查点。
        loaded = super(det3dCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded


class Writer:
    """训练日志记录器：写入 tensorboard 标量/文本并导出标量 JSON。"""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.log_mjson_file = None
        self.summary_writter = None
        self.metrics = []
        self._text_current_gstep = -1
        self._tb_texts = []

    def open(self):
        """打开 SummaryWriter。"""
        save_dir = self.save_dir
        assert save_dir.exists()
        summary_dir = save_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writter = SummaryWriter(str(summary_dir))
        return self

    def close(self):
        """导出标量 JSON 并关闭 SummaryWriter。"""
        assert self.summary_writter is not None
        tb_json_path = str(self.save_dir / "tensorboard_scalars.json")
        self.summary_writter.export_scalars_to_json(tb_json_path)
        self.summary_writter.close()
        self.summary_writter = None

    def log_text(self, text, step, tag="regular log"):
        """把文本加入 log.txt 与 tensorboard texts。"""
        if step > self._text_current_gstep and self._text_current_gstep != -1:
            # 跨 step 时，将累积文本批量写入 tensorboard 并清空缓存。
            total_text = "\n".join(self._tb_texts)
            self.summary_writter.add_text(tag, total_text, global_step=step)
            self._tb_texts = []
            self._text_current_gstep = step
        else:
            self._tb_texts.append(text)

        if self._text_current_gstep == -1:
            self._text_current_gstep = step

    def log_metrics(self, metrics: dict, step):
        """把扁平化后的指标以标量写入 tensorboard。"""
        flatted_summarys = flat_nested_json_dict(metrics, "/")
        for k, v in flatted_summarys.items():
            if isinstance(v, (list, tuple)):
                if any([isinstance(e, str) for e in v]):
                    continue
                v_dict = {str(i): e for i, e in enumerate(v)}
                for k1, v1 in v_dict.items():
                    self.summary_writter.add_scalar(k + "/" + k1, v1, step)
            else:
                if isinstance(v, str):
                    continue
                self.summary_writter.add_scalar(k, v, step)
