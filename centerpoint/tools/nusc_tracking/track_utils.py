"""跟踪匹配工具函数。

对应论文 Sec. 3.5 的贪心最近点匹配：给定 (检测, 轨迹) 距离矩阵，逐行取最近的
未占用列完成一对一匹配；距离极大（无效）的配对自动跳过。

主要函数：
    - greedy_assignment: 贪心最近点匹配。

与其他模块关系：
    - 被 pub_tracker.py 调用。
"""

import numpy as np

def greedy_assignment(dist):
  """贪心最近点匹配。

  Args:
      dist (ndarray): (N检测, M轨迹) 的距离矩阵，无效配对已被置为极大值(1e18)。

  Returns:
      ndarray: 形状 (K, 2) 的匹配索引 [检测索引, 轨迹索引]。
  """
  matched_indices = []
  if dist.shape[1] == 0:
    # 没有候选轨迹列，直接返回空匹配。
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    # 每个检测取距离最近的轨迹列。
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      # 该轨迹列已被占用，置为极大值防止重复匹配。
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
