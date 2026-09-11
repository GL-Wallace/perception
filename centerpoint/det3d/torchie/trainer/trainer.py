"""训练引擎核心：Trainer 类及其辅助组件。

Trainer 把模型、优化器、数据加载器与各类 Hook 组织成完整的训练/验证流程：
按 workflow 依次运行 train/val 阶段，在每个 epoch/iter 的固定时机触发 Hook
回调，同时维护日志缓冲、checkpoint 保存与恢复。

主要组件：
    - Trainer: 训练主循环的驱动类，负责注册并按优先级调度 Hook。
    - example_to_device: 将单个 batch 的样本字典搬运到 GPU。
    - parse_second_losses: 把 SECOND 风格的多项损失整理为总损失与日志变量。
    - BackgroundGenerator / Prefetcher: 数据预取辅助类（当前训练循环未启用）。

依赖 trainer.hooks 提供各类 Hook，依赖 checkpoint / log_buffer / priority /
utils 提供检查点读写、日志缓冲、优先级解析与分布式工具函数。
"""

import logging
import os.path as osp
import queue
import sys
import threading
import time
from collections import OrderedDict

import torch
from det3d import torchie

from . import hooks
from .checkpoint import load_checkpoint, save_checkpoint
from .hooks import (
    CheckpointHook,
    Hook,
    IterTimerHook,
    LrUpdaterHook,
    OptimizerHook,
    lr_updater,
)
from .log_buffer import LogBuffer
from .priority import get_priority
from .utils import (
    all_gather,
    get_dist_info,
    get_host_info,
    get_time_str,
    obj_from_dict,
    synchronize,
)


def example_to_device(example, device, non_blocking=False) -> dict:
    """将单个 batch 的样本字典搬运到指定设备。

    根据字段名区分数据形态：列表型字段逐元素搬运，张量型字段整体搬运，
    calib 字段则递归搬运其内部各标定张量，其余字段原样保留。

    Args:
        example (dict): 数据加载器产出的单 batch 字典，键为字段名。
        device (torch.device 或 int): 目标设备。
        non_blocking (bool): 是否采用非阻塞异步拷贝。

    Returns:
        dict: 搬运到目标设备后的样本字典。
    """
    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm",
                "anno_box", "ind", "mask", 'cat', 'points']:
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
            "cyv_num_points",
            "gt_boxes_and_cls"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = v1.to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_second_losses(losses):
    """整理 SECOND 风格模型的损失输出。

    对 losses["loss"] 求和得到总损失；其余每个键的值转换为 Python 标量组成的
    列表，其中 loc_loss_elem 为嵌套列表，其余为单层列表，便于后续日志记录。

    Args:
        losses (dict): 模型返回的损失字典，值为张量或张量列表。

    Returns:
        tuple: (总损失 Tensor, 待记录日志的变量 OrderedDict)。
    """
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


class BackgroundGenerator(threading.Thread):
    """在后台线程中预取生成器的数据。

    通过线程与有界队列实现预取，使主线程取数与生成器迭代解耦，
    从而减少训练过程中的数据加载等待。当前训练循环未启用该预取方式。

    Args:
        generator (iterable): 需要预取的生成器/可迭代对象。
        max_prefetch (int): 队列最大缓冲量，即最大预取个数。
    """

    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        # 生成器耗尽后向队列写入 None 作为结束哨兵
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 兼容：实现 __next__ 以支持 next() 内置函数
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class Prefetcher(object):
    """在独立 CUDA stream 上预取并搬运下一个 batch。

    提前把下一个 batch 拷贝到 GPU，主 stream 仅需等待该拷贝完成即可取用，
    从而隐藏 CPU 数据加载与 Host-to-Device 传输延迟。当前训练循环未启用。

    Args:
        dataloader (DataLoader): 待预取的数据加载器。
    """

    def __init__(self, dataloader):
        self.loader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # 在当前预取 stream 中完成设备搬运，与主 stream 的转发并行进行
        with torch.cuda.stream(self.stream):
            self.next_input = example_to_device(
                self.next_input, torch.cuda.current_device(), non_blocking=False
            )

    def next(self):
        # 等待预取 stream 上的拷贝完成后，再交给主 stream 使用
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        self.preload()
        return input


class Trainer(object):
    """PyTorch 训练辅助类，驱动训练/验证主循环。

    负责组装模型、优化器、数据加载器与 Hook，按 workflow 依次运行
    train/val 阶段，并在每个 epoch/iter 的关键时机触发 Hook 回调。

    Args:
        model (nn.Module): 待训练的模型。
        batch_processor (callable): 处理单 batch 前向与损失计算的函数。
        optimizer (torch.optim.Optimizer 或 dict, 可选): 优化器。
        lr_scheduler (可选): 学习率调度器。
        work_dir (str, 可选): 保存日志与 checkpoint 的工作目录。
        log_level (int): 日志级别，默认 logging.INFO。
        logger (logging.Logger, 可选): 外部传入的日志器。
    """

    def __init__(
        self,
        model,
        batch_processor,
        optimizer=None,
        lr_scheduler=None,
        work_dir=None,
        log_level=logging.INFO,
        logger=None,
        **kwargs,
    ):
        assert callable(batch_processor)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.batch_processor = batch_processor

        # 创建工作目录
        if torchie.is_str(work_dir):
            self.work_dir = osp.abspath(work_dir)
            torchie.mkdir_or_exist(self.work_dir)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError("'work_dir' must be a str or None")

        # 从模型类名获取模型名（分布式包装下取被包装的真实类名）
        if hasattr(self.model, "module"):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        # 记录当前进程在分布式训练中的 rank 与进程总数
        self._rank, self._world_size = get_dist_info()
        self.timestamp = get_time_str()
        if logger is None:
            self.logger = self.init_logger(work_dir, log_level)
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()

        self.mode = None
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: 模型名，通常为模型类名（分布式下为被包装类的类名）。"""
        return self._model_name

    @property
    def rank(self):
        """int: 当前进程在分布式训练中的 rank。"""
        return self._rank

    @property
    def world_size(self):
        """int: 参与训练的进程总数。"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: 已注册的 Hook 列表（按优先级升序排列）。"""
        return self._hooks

    @property
    def epoch(self):
        """int: 当前 epoch。"""
        return self._epoch

    @property
    def iter(self):
        """int: 当前全局迭代数。"""
        return self._iter

    @property
    def inner_iter(self):
        """int: 当前 epoch 内的迭代数。"""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: 最大训练 epoch 数。"""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: 最大训练迭代数。"""
        return self._max_iters

    def init_optimizer(self, optimizer):
        """初始化优化器。

        Args:
            optimizer (dict 或 :obj:`~torch.optim.Optimizer`): 优化器配置
                （含 type 键）或已实例化的优化器。

        Returns:
            :obj:`~torch.optim.Optimizer`: 初始化好的优化器。

        Examples:
            >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
            >>> type(runner.init_optimizer(optimizer))
            <class 'torch.optim.sgd.SGD`>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(
                optimizer, torch.optim, dict(params=self.model.parameters())
            )
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be either an Optimizer object or a dict, "
                "but got {}".format(type(optimizer))
            )
        return optimizer

    def _add_file_handler(self, logger, filename=None, mode="w", level=logging.INFO):
        """为日志器挂载一个文件输出处理器。

        Args:
            logger (logging.Logger): 目标日志器。
            filename (str): 日志文件路径。
            mode (str): 文件打开模式，默认以写入方式覆盖。
            level (int): 该文件处理器记录的最低日志级别。

        Returns:
            logging.Logger: 挂载完成后的日志器。
        """
        # TODO: move this method out of runner
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return logger

    def init_logger(self, log_dir=None, level=logging.INFO):
        """初始化日志器。

        仅 rank 0 进程会将日志写入工作目录下的时间戳命名文件，避免多进程重复写。

        Args:
            log_dir (str, 可选): 日志文件输出目录。
            level (int): 日志级别。

        Returns:
            :obj:`~logging.Logger`: Python 日志器。
        """
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - % (message)s", level=level
        )
        logger = logging.getLogger(__name__)
        if log_dir and self.rank == 0:
            filename = "{}.log".format(self.timestamp)
            log_file = osp.join(log_dir, filename)
            self._add_file_handler(logger, log_file, level=level)
        return logger

    def current_lr(self):
        """获取优化器各参数组的当前学习率。

        Returns:
            list[float]: 每个参数组对应的学习率。
        """
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def register_hook(self, hook, priority="NORMAL"):
        """将 Hook 注册到 Hook 列表中。

        Hook 列表始终按优先级升序排列（数值越小优先级越高），
        call_hook 调用时即按此顺序触发。

        Args:
            hook (:obj:`Hook`): 需注册的 Hook 实例。
            priority (int 或 str 或 :obj:`Priority`): Hook 优先级。
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # 从后向前找到第一个优先级不高于当前的位置，插入其后以保持升序
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            # 优先级比所有已有 Hook 都高，插入到列表最前
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        """从配置构造 Hook 实例。

        Args:
            args (Hook 或 dict): 已实例化的 Hook，或用于构造的配置字典。
            hook_type (type): args 为 dict 时必须指定，为 Hook 的子类。

        Returns:
            Hook: 构造得到的 Hook 实例。
        """
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError(
                "'args' must be either a Hook object"
                " or dict, not {}".format(type(args))
            )

    def call_hook(self, fn_name):
        """按优先级顺序触发所有 Hook 的同名钩子方法。

        Args:
            fn_name (str): 要调用的钩子方法名（如 before_train_iter）。
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        """加载 checkpoint 到当前模型。

        Args:
            filename (str): checkpoint 文件路径。
            map_location (str): 与 :func:`torch.load` 相同的设备映射参数。
            strict (bool): 是否严格要求参数完全匹配。

        Returns:
            dict 或 OrderedDict: 加载得到的 checkpoint。
        """
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(
        self, out_dir, filename_tmpl="epoch_{}.pth", save_optimizer=True, meta=None
    ):
        """保存当前模型 checkpoint，并创建 latest.pth 软链接。

        Args:
            out_dir (str): 输出目录。
            filename_tmpl (str): 文件名模板，最终以当前 epoch+1 填充。
            save_optimizer (bool): 是否一并保存优化器状态。
            meta (dict, 可选): 额外元信息，最终会补充 epoch 与 iter。
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        linkpath = osp.join(out_dir, "latest.pth")
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # 使用相对路径软链接指向最新 checkpoint
        torchie.symlink(filename, linkpath)

    def batch_processor_inline(self, model, data, train_mode, **kwargs):
        """内联版的单 batch 处理：数据搬运、前向、损失整理。

        相比外部 batch_processor，它在关键步骤之间插入 Hook 触发点，
        便于计时（transfer/forward/loss_parse）。

        Args:
            model (nn.Module): 模型。
            data (dict): 单 batch 数据。
            train_mode (bool): 是否为训练模式。
            **kwargs: 透传的额外参数。

        Returns:
            dict: 包含 loss、log_vars、num_samples 的输出字典；
            验证模式下直接返回模型推理结果。
        """
        if "local_rank" in kwargs:
            device = torch.device(kwargs["local_rank"])
        else:
            device = None

        # data = example_convert_to_torch(data, device=device)
        example = example_to_device(
            data, torch.cuda.current_device(), non_blocking=False
        )

        # 数据完成设备搬运后，通知计时 Hook 记录 transfer_time
        self.call_hook("after_data_to_device")

        if train_mode:
            losses = model(example, return_loss=True)
            # 前向完成后，通知计时 Hook 记录 forward_time
            self.call_hook("after_forward")
            loss, log_vars = parse_second_losses(losses)
            del losses

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=-1  # TODO: FIX THIS
            )
            # 损失解析完成后，通知计时 Hook 记录 loss_parse_time
            self.call_hook("after_parse_loss")

            return outputs
        else:
            return model(example, return_loss=False)

    def train(self, data_loader, epoch, **kwargs):
        """运行一个训练 epoch。

        在单个 epoch 内逐批迭代数据；每批前后分别触发 before_train_iter /
        after_train_iter 等 Hook，并推进学习率调度器与全局迭代计数。

        Args:
            data_loader (DataLoader): 本 epoch 使用的数据加载器。
            epoch (int): 当前 epoch 编号（用于计算全局 step）。
            **kwargs: 透传给 batch 处理函数的参数。
        """
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self.length = len(data_loader)
        # 全局最大迭代数为总 epoch 数乘以每 epoch 的批次数
        self._max_iters = self._max_epochs * self.length
        self.call_hook("before_train_epoch")

        base_step = epoch * self.length

        # prefetcher = Prefetcher(data_loader)
        # for data_batch in BackgroundGenerator(data_loader, max_prefetch=3):
        for i, data_batch in enumerate(data_loader):
            global_step = base_step + i
            if self.lr_scheduler is not None:
                #print(global_step)
                self.lr_scheduler.step(global_step)

            self._inner_iter = i

            self.call_hook("before_train_iter")

            # outputs = self.batch_processor(self.model,
            #                                data_batch,
            #                                train_mode=True,
            #                                **kwargs)
            outputs = self.batch_processor_inline(
                self.model, data_batch, train_mode=True, **kwargs
            )

            if not isinstance(outputs, dict):
                raise TypeError("batch_processor() must return a dict")
            # 将本批日志变量送入 LogBuffer，供 LoggerHook 按周期取平均并输出
            if "log_vars" in outputs:
                self.log_buffer.update(outputs["log_vars"], outputs["num_samples"])
            self.outputs = outputs
            self.call_hook("after_train_iter")
            self._iter += 1

        self.call_hook("after_train_epoch")
        self._epoch += 1

    def val(self, data_loader, **kwargs):
        """运行一个验证 epoch。

        逐批推理并将预测搬运回 CPU，按样本 token 汇总；随后通过 all_gather
        聚合各 rank 的预测，最后仅 rank 0 执行数据集评估并打印结果。

        Args:
            data_loader (DataLoader): 验证集数据加载器。
            **kwargs: 透传给 batch 处理函数的参数。
        """
        self.model.eval()
        self.mode = "val"
        self.data_loader = data_loader
        self.call_hook("before_val_epoch")

        self.logger.info(f"work dir: {self.work_dir}")

        if self.rank == 0:
            prog_bar = torchie.ProgressBar(len(data_loader.dataset))

        detections = {}
        cpu_device = torch.device("cpu")

        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook("before_val_iter")
            with torch.no_grad():
                outputs = self.batch_processor(
                    self.model, data_batch, train_mode=False, **kwargs
                )
            for output in outputs:
                token = output["metadata"]["token"]
                # 除 metadata 外的张量结果统一搬回 CPU，避免占用显存
                for k, v in output.items():
                    if k not in [
                        "metadata",
                    ]:
                        output[k] = v.to(cpu_device)
                detections.update(
                    {token: output,}
                )
                if self.rank == 0:
                    # 每个样本在 world_size 个 rank 上各处理一次，进度条按总次数推进
                    for _ in range(self.world_size):
                        prog_bar.update()

        # 所有进程同步，确保等到各 rank 完成验证后再进行 all_gather
        synchronize()

        all_predictions = all_gather(detections)

        # 仅 rank 0 汇总并评估，其余 rank 直接返回
        if self.rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        # torch.save(predictions, "final_predictions_debug.pkl")
        # TODO fix evaluation module
        result_dict, _ = self.data_loader.dataset.evaluation(
            predictions, output_dir=self.work_dir
        )

        self.logger.info("\n")
        for k, v in result_dict["results"].items():
            self.logger.info(f"Evaluation {k}: {v}")

        self.call_hook("after_val_epoch")

    def resume(self, checkpoint, resume_optimizer=True, map_location="default"):
        """从 checkpoint 恢复训练状态（epoch、iter 及可选的优化器）。

        Args:
            checkpoint (str): checkpoint 文件路径。
            resume_optimizer (bool): 是否恢复优化器状态。
            map_location (str): 设备映射参数，默认加载到当前 CUDA 设备。
        """
        if map_location == "default":
            checkpoint = self.load_checkpoint(
                checkpoint , map_location='cuda:{}'.format(torch.cuda.current_device()) # TODO: FIX THIS!!
            )
        else:
            checkpoint = self.load_checkpoint(checkpoint, map_location=map_location)

        self._epoch = checkpoint["meta"]["epoch"]
        self._iter = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.iter)

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """启动训练主循环。

        按 workflow 中声明的 (阶段, epoch 数) 顺序循环执行各阶段，
        直到达到 max_epochs 为止。

        Args:
            data_loaders (list[:obj:`DataLoader`]): 与 workflow 一一对应的数据加载器。
            workflow (list[tuple]): 由 (phase, epochs) 组成的运行顺序，
                phase 为字符串方法名或可调用对象。
            max_epochs (int): 最大训练 epoch 数。
        """
        assert isinstance(data_loaders, list)
        assert torchie.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        work_dir = self.work_dir if self.work_dir is not None else "NONE"
        self.logger.info(
            "Start running, host: %s, work_dir: %s", get_host_info(), work_dir
        )
        self.logger.info("workflow: %s, max: %d epochs", workflow, max_epochs)
        self.call_hook("before_run")

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):
                    if not hasattr(self, mode):
                        raise ValueError(
                            "Trainer has no method named '{}' to run an epoch".format(
                                mode
                            )
                        )
                    epoch_runner = getattr(self, mode)
                elif callable(mode):
                    epoch_runner = mode
                else:
                    raise TypeError(
                        "mode in workflow must be a str or "
                        "callable function not '{}'".format(type(mode))
                    )

                for _ in range(epochs):
                    if mode == "train" and self.epoch >= max_epochs:
                        return
                    elif mode == "val":
                        # val 阶段由 epoch_runner 自行推进，无需显式传 epoch
                        epoch_runner(data_loaders[i], **kwargs)
                    else:
                        # train 阶段需要显式传入当前 epoch 以计算全局 step
                        epoch_runner(data_loaders[i], self.epoch, **kwargs)

        # time.sleep(1)
        self.call_hook("after_run")

    def register_lr_hooks(self, lr_config):
        """根据配置注册学习率调整 Hook。

        Args:
            lr_config (LrUpdaterHook 或 dict): 已实例化的 Hook 或配置字典，
                dict 需包含 policy 键以确定具体策略类型。
        """
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert "policy" in lr_config
            hook_name = lr_config["policy"].title() + "LrUpdaterHook"
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError(
                "'lr_config' must be eigher a LrUpdaterHook object"
                " or dict, not '{}'".format(type(lr_config))
            )

    def register_logger_hooks(self, log_config):
        """根据配置注册日志 Hook。

        将 log_config["interval"] 作为公共输出周期，依次实例化并注册
        log_config["hooks"] 中的各类日志 Hook（默认 VERY_LOW 优先级）。

        Args:
            log_config (dict): 含 interval 与 hooks 的日志配置。
        """
        log_interval = log_config["interval"]
        for info in log_config["hooks"]:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval)
            )
            self.register_hook(logger_hook, priority="VERY_LOW")

    def register_training_hooks(
        self, lr_config, optimizer_config=None, checkpoint_config=None, log_config=None
    ):
        """注册训练所需的默认 Hook。

        默认包含：
            - LrUpdaterHook：学习率调整。
            - OptimizerStepperHook：梯度清零、反向与参数更新。
            - CheckpointSaverHook：周期性保存 checkpoint。
            - IterTimerHook：各阶段耗时统计。
            - LoggerHook(s)：训练日志输出。
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        if lr_config is not None:
            assert self.lr_scheduler is None
            self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
