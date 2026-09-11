"""旋转 3D 框 IoU 与 NMS 算子子包。

导出 CUDA 扩展 iou3d_nms_cuda 及其 Python 封装 iou3d_nms_utils，
用于旋转 3D 框的 IoU 计算与非极大值抑制。
"""

from det3d.ops.iou3d_nms import iou3d_nms_cuda, iou3d_nms_utils
