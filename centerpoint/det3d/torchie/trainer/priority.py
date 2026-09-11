"""Hook 优先级定义与解析。

Hook 以整数优先级排序（数值越小越先触发），本模块提供常用优先级枚举与
将 int / str / Priority 统一转换为数值的工具，供 Trainer.register_hook 使用。

主要类/函数：
    - Priority: 定义 HIGHEST 到 LOWEST 的常用优先级枚举。
    - get_priority: 把各种优先级表示转换为整数值。
"""

from enum import Enum


class Priority(Enum):
    """Hook 优先级等级。

    +------------+------------+
    | 等级       | 数值       |
    +============+============+
    | HIGHEST    | 0          |
    +------------+------------+
    | VERY_HIGH  | 10         |
    +------------+------------+
    | HIGH       | 30         |
    +------------+------------+
    | NORMAL     | 50         |
    +------------+------------+
    | LOW        | 70         |
    +------------+------------+
    | VERY_LOW   | 90         |
    +------------+------------+
    | LOWEST     | 100        |
    +------------+------------+
    数值越小优先级越高，Hook 越早被触发。
    """

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    NORMAL = 50
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority):
    """把优先级表示转换为整数值。

    Args:
        priority (int 或 str 或 :obj:`Priority`): 优先级。

    Returns:
        int: 对应的优先级数值。
    """
    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError("priority must be between 0 and 100")
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError("priority must be an integer or Priority enum value")
