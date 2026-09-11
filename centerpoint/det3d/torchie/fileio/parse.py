"""文本文件解析工具。

提供将简单文本文件解析为列表或字典的辅助函数，常用于读取类别名列表、
评测标签映射等训练/评测配置相关的纯文本数据。

主要函数：
    - list_from_file: 把文本文件的每一行读成列表元素。
    - dict_from_file: 把每行按键值两列解析成字典。
"""


def list_from_file(filename, prefix="", offset=0, max_num=0):
    """读取文本文件并将内容解析为字符串列表。

    Args:
        filename (str): 文件名。
        prefix (str): 插入到每个元素开头的字符串前缀。
        offset (int): 跳过的起始行数。
        max_num (int): 最多读取的行数，0 或负数表示不限制。

    Returns:
        list[str]: 字符串列表。
    """
    cnt = 0
    item_list = []
    with open(filename, "r") as f:
        # 先按 offset 跳过若干行。
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip("\n"))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str):
    """读取文本文件并将内容解析为字典。

    文件的每一行会被空白或制表符切分成两列或多列：第一列解析为键，之后的列解析为值。

    Args:
        filename(str): 文件名。
        key_type(type): 字典键的类型，默认 str，指定后会对键做类型转换。

    Returns:
        dict: 解析出的内容。
    """
    mapping = {}
    with open(filename, "r") as f:
        for line in f:
            items = line.rstrip("\n").split()
            assert len(items) >= 2
            key = key_type(items[0])
            # 仅两列时值退化为单个字符串，多于两列时保留为列表。
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
