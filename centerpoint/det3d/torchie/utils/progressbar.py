"""进度条工具。

提供终端进度条 ProgressBar，以及串行、并行、迭代三种任务执行方式对应的进度跟踪
封装，常用于数据预处理（tools 脚本批量转换）与训练/评测流程中展示任务进度与
预估剩余时间（ETA）。

主要类/函数：
    - ProgressBar: 终端进度条。
    - track_progress: 串行执行任务并显示进度。
    - track_parallel_progress: 多进程池执行任务并显示进度。
    - track_iter_progress: 以生成器方式逐项产出任务并显示进度。
    - init_pool: 构造多进程池的内部辅助。
"""

import sys
from multiprocessing import Pool

from .misc import collections_abc
from .timer import Timer


class ProgressBar(object):
    """可在终端打印进度的进度条"""

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        # 条宽不超过终端可容纳的最大宽度。
        self.bar_width = bar_width if bar_width <= max_bar_width else max_bar_width
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        # 依据终端宽度计算一个合理的最大条宽，避免过长折行。
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(
                "terminal width is too small ({}), please consider "
                "widen the terminal for better progressbar "
                "visualization".format(terminal_width)
            )
            max_bar_width = 10
        return max_bar_width

    def start(self):
        """打印进度条初始行并启动计时。"""
        if self.task_num > 0:
            sys.stdout.write(
                "[{}] 0/{}, elapsed: 0s, ETA:".format(
                    " " * self.bar_width, self.task_num
                )
            )
        else:
            sys.stdout.write("completed: 0, elapsed: 0s")
        sys.stdout.flush()
        self.timer = Timer()

    def update(self):
        """完成一项任务后刷新进度、速率与剩余时间。"""
        self.completed += 1
        elapsed = self.timer.since_start()
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            # 按已完成比例估算剩余秒数 ETA。
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            # ">" 表示已完成部分，空格表示未完成部分。
            bar_chars = ">" * mark_width + " " * (self.bar_width - mark_width)
            sys.stdout.write(
                "\r[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s".format(
                    bar_chars,
                    self.completed,
                    self.task_num,
                    fps,
                    int(elapsed + 0.5),
                    eta,
                )
            )
        else:
            sys.stdout.write(
                "completed: {}, elapsed: {}s, {:.1f} tasks/s".format(
                    self.completed, int(elapsed + 0.5), fps
                )
            )
        sys.stdout.flush()


def track_progress(func, tasks, bar_width=50, **kwargs):
    """以进度条跟踪串行任务执行。

    任务通过简单的 for 循环依次执行。

    Args:
        func (callable): 应用到每个任务上的函数。
        tasks (list 或 tuple[Iterable, int]): 任务列表，或 (任务迭代器, 总数) 元组。
        bar_width (int): 进度条宽度。

    Returns:
        list: 各任务执行结果。
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    sys.stdout.write("\n")
    return results


def init_pool(process_num, initializer=None, initargs=None):
    """按参数构造 multiprocessing 进程池。"""
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(
    func,
    tasks,
    nproc,
    initializer=None,
    initargs=None,
    bar_width=50,
    chunksize=1,
    skip_first=False,
    keep_order=True,
):
    """以进度条跟踪多进程并行任务执行。

    使用内置 :mod:`multiprocessing` 模块创建进程池，任务通过
    :func:`Pool.map` 或 :func:`Pool.imap_unordered` 执行。

    Args:
        func (callable): 应用到每个任务上的函数。
        tasks (list 或 tuple[Iterable, int]): 任务列表，或 (任务迭代器, 总数) 元组。
        nproc (int): 进程（worker）数量。
        initializer (None 或 callable): 含义同 :class:`multiprocessing.Pool`。
        initargs (None 或 tuple): 含义同 :class:`multiprocessing.Pool`。
        chunksize (int): 含义同 :class:`multiprocessing.Pool`。
        bar_width (int): 进度条宽度。
        skip_first (bool): 估算速率时是否跳过每个 worker 的首个样本，
            因为首样本可能包含较慢的进程初始化。
        keep_order (bool): 为 True 时使用 :func:`Pool.imap`（保序），否则使用
            :func:`Pool.imap_unordered`（不保证顺序）。

    Returns:
        list: 各任务执行结果。
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    # 跳过首批样本时，从总数中扣除这部分，进度条也从稍后开始。
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            # 首批样本（每个 worker 各 chunksize 个）不更新进度，待凑齐后再开始。
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    sys.stdout.write("\n")
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks, bar_width=50, **kwargs):
    """以进度条跟踪任务迭代（生成器）进度。

    任务通过简单的 for 循环逐项产出。

    Args:
        tasks (list 或 tuple[Iterable, int]): 任务列表，或 (任务迭代器, 总数) 元组。
        bar_width (int): 进度条宽度。

    Yields:
        list: 逐项产出的任务结果。
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], collections_abc.Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, collections_abc.Iterable):
        task_num = len(tasks)
    else:
        raise TypeError('"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width)
    for task in tasks:
        yield task
        prog_bar.update()
    sys.stdout.write("\n")
