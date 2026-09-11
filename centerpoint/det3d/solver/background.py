"""后台数据预取生成器。

通过独立线程在后台提前生成数据并放入队列，减少主线程等待数据加载的时间，
从而提升训练数据吞吐。
"""

import threading, queue


class BackgroundGenerator(threading.Thread):
    """后台线程预取生成器，将源生成器的输出提前缓存到队列中。

    Args:
        generator: 源生成器。
        max_prefetch: 队列最大预取数量。
    """

    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        # 作为守护线程，主线程退出时随之结束。
        self.daemon = True
        self.start()

    def run(self):
        """后台线程主体：持续从源生成器取数据放入队列，结束时放入 None 哨兵。"""
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        """取出下一个数据项；遇到 None 哨兵则抛 StopIteration。"""
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 兼容
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
