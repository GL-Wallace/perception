"""PointPillars 检测器：基于柱（pillar）表示的单阶段检测器。

PointPillars 继承 SingleStageDetector，先由 reader（PillarFeatureNet）对体素化
的柱特征做编码，再由 backbone（PointPillarsScatter）把柱特征散布回 BEV 伪图像，
经 neck 与 bbox_head 输出检测结果。

对应论文 Sec 3：PointPillars 是文中讨论的标准 3D backbone 之一。
"""

from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    """基于柱表示的单阶段检测器，继承自 SingleStageDetector。"""

    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        """构造 PointPillars，直接复用父类组装 reader/backbone/neck/bbox_head。

        Args:
            reader (dict): 柱特征编码 reader（PillarFeatureNet）配置。
            backbone (dict): 柱散布 backbone（PointPillarsScatter）配置。
            neck (dict): BEV neck 配置。
            bbox_head (dict): 检测头配置。
            train_cfg (dict, optional): 训练配置。
            test_cfg (dict, optional): 测试配置。
            pretrained (str, optional): 预训练权重路径。
        """
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        """提取柱特征并执行 backbone + neck。

        Args:
            data: 含 features/num_voxels/coors/batch_size/input_shape 的字典。

        Returns:
            经 neck 处理后的 BEV 特征张量。
        """
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        """单阶段前向：组装输入字典 → 提取特征 → bbox_head 预测 → 损失/解码。

        从 example 中取出已体素化的数据，构造 reader/backbone 所需输入字典。

        Args:
            example: 含体素化数据（voxels/coordinates/num_points/num_voxels）的输入。
            return_loss (bool): True 时计算损失。

        Returns:
            损失（训练）或预测结果（推理）。
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        """两阶段前向：返回一阶段预测框、BEV 特征与一阶段损失。

        供 TwoStageDetector 调用，输出交给第二阶段 roi_head 做精炼。

        Args:
            example: 输入数据。
            return_loss (bool): True 时返回一阶段损失。

        Returns:
            训练返回 (boxes, bev_feature, loss)；推理返回 (boxes, bev_feature, None)。
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds, _ = self.bbox_head(x)

        # 手动深拷贝：detach 出无梯度的一阶段预测副本，供解码框使用，避免梯度回传。
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
