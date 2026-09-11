"""打印辅助工具。

提供嵌套 dict 展平与指标格式化为字符串的函数，供日志输出使用。
"""


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    """递归展平嵌套 dict。

    Args:
        json_dict: 待展平的 dict。
        flatted: 保存展平结果的 dict（原地修改）。
        sep: 层级分隔符。
        start: 键前缀。
    """
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + str(k))
        else:
            flatted[start + sep + str(k)] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """把嵌套 json 风格 dict 展平为单层 dict（浅拷贝）。

    Args:
        json_dict: 待展平的嵌套 dict。
        sep: 层级分隔符。

    Returns:
        dict: 展平后的单层 dict。
    """
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

    Args:
        metrics: 指标 dict（可嵌套）。
        sep: 展平时的层级分隔符。

    Returns:
        str: 格式化后的指标字符串。
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
