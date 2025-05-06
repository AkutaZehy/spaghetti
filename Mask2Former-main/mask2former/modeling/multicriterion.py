from .criterion import SetCriterion
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

class MultiTaskSetCriterion(SetCriterion):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: dict,
        eos_coef: float,
        losses: list,
        num_types: int,
        num_positions: int,
        type_weight: float,
        position_weight: float,
        num_points: int = 0,
        oversample_ratio: float = 0.7,
        importance_sample_ratio: float = 0.75,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=losses,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio
        )
        
        # 初始化多任务参数
        self.num_types = num_types
        self.num_positions = num_positions
        
        # 正确扩展权重字典
        self.weight_dict = weight_dict.copy()  # 避免修改原始字典
        self.weight_dict.update({
            "loss_type": type_weight,
            "loss_position": position_weight
        })
        
        # 用于三个任务的背景类权重向量
        self.empty_weight = torch.ones(self.num_classes + 1)
        self.empty_weight[-1] = self.eos_coef

        self.type_empty_weight = torch.ones(self.num_types + 1)
        self.type_empty_weight[-1] = self.eos_coef

        self.position_empty_weight = torch.ones(self.num_positions + 1)
        self.position_empty_weight[-1] = self.eos_coef


    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # 获取主分支匹配
        indices = self.matcher(outputs_without_aux, targets)

        # Normalization
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # 主损失
        losses = {}
        
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # 辅助分支损失
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    # 保证 loss 名字唯一（避免被覆盖）
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # print(idx, target_classes_o, "TARGETO+LAB")

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # 需要 [N, C, T] 格式
            target_classes,
            # torch.ones(self.num_classes + 1, device=src_logits.device)
            weight=self.empty_weight
        )
        
        losses = {"loss_ce": loss_ce}
        return losses
      
    def loss_types(self, outputs, targets, indices, num_masks):
        """类型分类损失（交叉熵）"""
        assert "pred_types" in outputs, "输出中必须包含 pred_types 字段"
        src_types = outputs["pred_types"].float()  # 形状 [batch_size, num_queries, num_types]

        # 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        
        # 提取目标类型标签
        target_types = torch.cat(
            [t["type_ids"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )
        
        # print(idx, target_types, "TARGETO")
        
        # 填充默认值（背景类）
        target_classes = torch.full(
            src_types.shape[:2], 
            self.num_types,
            dtype=torch.int64, 
            device=src_types.device
        )
        target_classes[idx] = target_types  # 填充匹配的位置
        
        # print(idx, target_classes, "TARGETO")
        # print(src_types.transpose(1,2), "SRC")
        # print(self.type_empty_weight.to(src_types.device), "=CHECKWEIGHT")

        # 计算交叉熵损失
        loss_type = F.cross_entropy(
            src_types.transpose(1, 2),  # 需要 [N, C, T] 格式
            target_classes,
            # torch.ones(self.num_types + 1, device=src_types.device)
            weight=self.type_empty_weight.to(src_types.device)
        )
        
        # print(loss_type, "=LOSS")
        
        return {"loss_type": loss_type}

    def loss_positions(self, outputs, targets, indices, num_masks):
        """位置分类损失（交叉熵）"""
        assert "pred_positions" in outputs, "输出中必须包含 pred_positions 字段"
        src_positions = outputs["pred_positions"].float()  # 形状 [batch_size, num_queries, num_positions]

        # 获取匹配的索引
        idx = self._get_src_permutation_idx(indices)
        
        # 提取目标位置标签
        target_positions = torch.cat(
            [t["position_ids"][J] for t, (_, J) in zip(targets, indices)], dim=0
        )
        
        # 填充默认值（背景类）
        target_classes = torch.full(
            src_positions.shape[:2], 
            self.num_positions,
            dtype=torch.int64, 
            device=src_positions.device
        )
        target_classes[idx] = target_positions  # 填充匹配的位置

        # 计算交叉熵损失
        loss_position = F.cross_entropy(
            src_positions.transpose(1, 2),  # 需要 [N, C, T] 格式
            target_classes,
            # torch.ones(self.num_positions + 1, device=src_positions.device)
            weight=self.position_empty_weight.to(src_positions.device)
        )
        
        return {"loss_position": loss_position}
      
    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'types': self.loss_types,
            'positions': self.loss_positions,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)