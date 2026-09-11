"""计时工具。

提供轻量的 Timer 类与全局 check_time 计时点，用于测量训练循环、数据加载等
关键路径的耗时，辅助性能分析（如进度条计算 elapsed/ETA 时会复用 Timer）。

主要类/函数：
    - TimerError: 计时器相关异常。
    - Timer: 支持上下文管理器与分段计时的计时器。
    - check_time: 基于全局注册表的单行计时点。
"""

from time import time


class TimerError(Exception):
    """计时器在未运行时被查询等非法操作时抛出的异常。"""

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer(object):
    """一个灵活的计时器。

    支持 ``with`` 上下文管理器，以及自计时开始/自上次检查以来两种计时口径：
    可以在任意位置调用 since_start / since_last_check 记录分段耗时。

    :Example:

    >>> import time
    >>> import mmcv
    >>> with mmcv.Timer():
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    1.000
    >>> with mmcv.Timer(print_tmpl='it takes {:.1f} seconds'):
    >>>     # simulate a code block that will run for 1s
    >>>     time.sleep(1)
    it takes 1.0 seconds
    >>> timer = mmcv.Timer()
    >>> time.sleep(0.5)
    >>> print(timer.since_start())
    0.500
    >>> time.sleep(0.5)
    >>> print(timer.since_last_check())
    0.500
    >>> print(timer.since_start())
    1.000
    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else "{:.3f}"
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: 计时器是否正在运行"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        # 退出上下文时按模板打印本段耗时。
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """启动计时器。"""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        # 每次 start 都刷新「上次检查」时间点。
        self._t_last = time()

    def since_start(self):
        """自计时器启动以来的总时长。

        Returns (float): 以秒为单位的时间。
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """自上一次检查以来的时长。

        :func:`since_start` 与 :func:`since_last_check` 都属于检查操作，
        都会刷新「上次检查」时间点。

        Returns (float): 以秒为单位的时间。
        """
        if not self._is_running:
            raise TimerError("timer is not running")
        dur = time() - self._t_last
        self._t_last = time()
        return dur


_g_timers = {}  # global timers


def check_time(timer_id):
    """以单行方式添加计时检查点。

    适合对一批任务逐项计时：第一次调用时会为该 id 注册一个 Timer 并返回 0，
    此后每次调用返回自上次检查以来经过的秒数。

    :Example:

    >>> import time
    >>> import mmcv
    >>> for i in range(1, 6):
    >>>     # simulate a code block
    >>>     time.sleep(i)
    >>>     mmcv.check_time('task1')
    2.000
    3.000
    4.000
    5.000

    Args:
        timer_id (str): 计时器标识。
    """
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].since_last_check()
