"""nuScenes 跟踪器（CenterPoint 贪心最近点匹配实现）。

对应论文 Sec. 3.5：把多目标跟踪简化为贪心最近点匹配——利用第一阶段回归的速度
(vx, vy) 把上一帧对象中心平移到当前帧的预测位置，再与当前帧检测结果按类别与
距离做最近点关联；未匹配的检测生成新轨迹，连续多帧未匹配的旧轨迹按速度继续外推。

主要类：
    - PubTracker: 维护轨迹列表并逐帧执行速度平移与最近点匹配。

与其他模块关系：
    - 被 tools/nusc_tracking/pub_test.py 逐帧调用 step_centertrack；
    - 依赖 tools/nusc_tracking/track_utils.py 的 greedy_assignment 做贪心匹配，
      可选使用 scipy 的 linear_sum_assignment 做匈牙利匹配。
"""

import numpy as np
import copy
from track_utils import greedy_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import copy 
import importlib
import sys 

# nuScenes 需要参与跟踪评估的类别。
NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 每个类别 L2 速度误差分布的 99.9 分位数（单位：米 / 0.5 秒），
# 作为最近点匹配的距离上限：超过该距离视为不能匹配。
# 注意这是较早的统计值，未精细调参；针对自己的模型调优可带来可观的 AMOTA 提升。
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}



class PubTracker(object):
  """nuScenes 的贪心最近点跟踪器（可选匈牙利匹配）。"""

  def __init__(self,  hungarian=False, max_age=0):
    """初始化跟踪器并清空轨迹。

    Args:
        hungarian (bool): 是否用匈牙利算法做全局最优匹配（默认贪心）。
        max_age (int): 轨迹丢失后允许保留的最大帧数。
    """
    self.hungarian = hungarian
    self.max_age = max_age

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    self.reset()
  
  def reset(self):
    """重置轨迹与 ID 计数，用于每个新视频序列开始前。"""
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    """处理一帧检测结果，完成速度平移、最近点匹配与轨迹更新。

    论文 Sec. 3.5 贪心最近点匹配的核心实现：
        1) 检测中心 ct 为 BEV 二维坐标 (x, y)；
        2) 速度平移：tracking = velocity * (-time_lag)，把上一帧轨迹中心平移到
           当前帧的预测位置 dets = ct + tracking；
        3) 在 (轨迹, 检测) 之间算欧氏距离，类别不同或超过类别距离阈值的视为
           无效（距离置为极大值），再用贪心/匈牙利匹配；
        4) 未匹配检测分配新 ID；未匹配旧轨迹年龄未超 max_age 时按速度外推并保留。

    Args:
        results (list[dict]): 当前帧检测结果，含 translation/velocity/detection_name 等。
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
        if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
          continue 

        # ct 为 BEV 上的 2D 中心；tracking 为按速度与时间外推的位移（方向取反）。
        det['ct'] = np.array(det['translation'][:2])
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
        det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
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

    # 每个检测按其类别取对应的距离阈值（速度误差 99.9 分位数）。
    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

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
      if self.hungarian:
        dist[dist > 1e18] = 1e18
        matched_indices = np.array(linear_assignment(copy.deepcopy(dist)))
        matched_indices = matched_indices.transpose()
      else:
        # 贪心：每个检测依次取最近的未占用轨迹（见 track_utils.greedy_assignment）。
        matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # 首帧：没有既有轨迹可匹配
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.hungarian:
      # 匈牙利可能匹配到距离为极大值的配对，需拆回未匹配集合。
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    # 匹配成功：沿用旧轨迹 ID，刷新活跃帧数。
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    # 未匹配检测：视为新出现的对象，分配新 ID。
    for i in unmatched_dets:
      track = results[i]
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
