"""Waymo 跟踪器（CenterPoint 贪心最近点匹配实现）。

对应论文 Sec. 3.5：利用第一阶段回归的速度 (vx, vy) 把上一帧对象中心平移到当前
帧的预测位置，再与当前帧检测结果按类别与距离做贪心最近点匹配；未匹配且分数超
阈值的检测生成新轨迹，连续多帧未匹配的旧轨迹按速度外推保留。

与 nuScenes 版 pub_tracker 的区别：Waymo 版新轨迹额外受 score_thresh 过滤，
且距离阈值按类别通过 max_dist 参数传入。

主要类：
    - PubTracker: 维护轨迹列表并逐帧执行速度平移与贪心匹配。

与其他模块关系：
    - 被 tools/waymo_tracking/test.py 逐帧调用 step_centertrack。
"""

import numpy as np
import copy
import copy 
import importlib
import sys 

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


# Waymo 需要参与跟踪评估的类别。
WAYMO_TRACKING_NAMES = [
    'VEHICLE',
    'PEDESTRIAN',
    'CYCLIST'
]

class PubTracker(object):
  """Waymo 的贪心最近点跟踪器。"""

  def __init__(self, max_age=0, max_dist={}, score_thresh=0.1):
    """初始化跟踪器并清空轨迹。

    Args:
        max_age (int): 轨迹丢失后允许保留的最大帧数。
        max_dist (dict): 各类别的最近点匹配距离上限。
        score_thresh (float): 新轨迹（未匹配检测）的分数阈值。
    """
    self.max_age = max_age

    # 各类别距离阈值（最近点匹配上限），与 nuScenes 版的 VELOCITY_ERROR 作用相同。
    self.WAYMO_CLS_VELOCITY_ERROR = max_dist 

    self.WAYMO_TRACKING_NAMES = WAYMO_TRACKING_NAMES
    self.score_thresh = score_thresh 

    self.reset()
  
  def reset(self):
    """重置轨迹与 ID 计数，用于每个新视频序列开始前。"""
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    """处理一帧检测结果，完成速度平移、贪心最近点匹配与轨迹更新。

    论文 Sec. 3.5 贪心最近点匹配的核心实现：检测中心 ct 加上速度平移
    tracking = velocity * (-time_lag) 得到当前帧预测位置，与上一帧轨迹中心
    计算欧氏距离，按类别与距离阈值过滤后贪心匹配。

    Args:
        results (list[dict]): 当前帧检测结果，含 translation/velocity/detection_name/score。
        time_lag (float): 距上一帧的时间间隔（秒）。

    Returns:
        list[dict]: 更新后的轨迹列表（含 tracking_id/age/active 等字段）。
    """
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in results:
        # 过滤掉不参与跟踪评估的类别。
        if det['detection_name'] not in self.WAYMO_TRACKING_NAMES:
          print("filter {}".format(det['detection_name']))
          continue 

        # ct 为 BEV 上的 2D 中心；tracking 为按速度与时间外推的位移（方向取反）。
        det['ct'] = np.array(det['translation'][:2])
        det['tracking'] = np.array(det['velocity'][:2]) * -1 *  time_lag
        det['label_preds'] = self.WAYMO_TRACKING_NAMES.index(det['detection_name'])
        temp.append(det)

      results = temp

    N = len(results)
    M = len(self.tracks)

    # 检测的当前帧预测位置：中心 + 速度平移（论文 Sec. 3.5 的速度平移）。
    if 'tracking' in results[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in results], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in results], np.float32) 

    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    # 每个检测按其类别取对应的距离阈值。
    max_diff = np.array([self.WAYMO_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    # 上一帧轨迹中心（未做速度平移前的记录位置）。
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    if len(tracks) > 0:  # 非首帧
      # 轨迹中心与预测位置之间的欧氏距离矩阵 (N x M)。
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # 距离单位为米

      # 类别不一致或距离超过类别阈值的配对标记为无效。
      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      # 无效配对的距离置为极大值，从而不会在匹配中被选中。
      dist = dist  + invalid * 1e18
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # 首帧：没有既有轨迹可匹配
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    matches = matched_indices

    ret = []
    # 匹配成功：沿用旧轨迹 ID，刷新活跃帧数。
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    # 未匹配检测：分数超过阈值才视为新对象并分配新 ID。
    for i in unmatched_dets:
      track = results[i]
      if track['score'] > self.score_thresh:
        self.id_count += 1
        track['tracking_id'] = self.id_count
        track['age'] = 1
        track['active'] =  1
        ret.append(track)

    # 未匹配旧轨迹：年龄未超过 max_age 时保留（但不输出到当前帧），
    # 并按其速度继续向前外推中心位置，等待后续帧重新匹配。
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        track['age'] += 1
        track['active'] = 0
        ct = track['ct']

        # 用记录的速度位移向前外推（move forward）。
        if 'tracking' in track:
            offset = track['tracking'] * -1 # 向前移动
            track['ct'] = ct + offset 
        ret.append(track)

    self.tracks = ret
    return ret
