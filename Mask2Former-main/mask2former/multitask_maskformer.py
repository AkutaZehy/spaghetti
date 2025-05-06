import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from .maskformer_model import MaskFormer
from .modeling.multicriterion import MultiTaskSetCriterion
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, polygons_to_bitmask
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from .modeling.multimatcher import HungarianMatcher

@META_ARCH_REGISTRY.register()
class MultiTaskMaskFormer(MaskFormer):
    @classmethod
    def from_config(cls, cfg):
        # 获取原始配置
        base_config = super().from_config(cfg)
        
        # 读取所有权重参数
        weight_dict = {
            "loss_ce": cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            "loss_mask": cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            "loss_dice": cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            "loss_type": cfg.MODEL.MASK_FORMER.TYPE_WEIGHT,
            "loss_position": cfg.MODEL.MASK_FORMER.POSITION_WEIGHT
        }

        # 深度监督扩展
        if cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            
        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        type_weight = cfg.MODEL.MASK_FORMER.TYPE_WEIGHT
        position_weight = cfg.MODEL.MASK_FORMER.POSITION_WEIGHT
        
        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        # 构建多任务Criterion
        base_config["criterion"] = MultiTaskSetCriterion(
            num_classes=base_config["sem_seg_head"].num_classes,
            num_types=cfg.MODEL.SEM_SEG_HEAD.NUM_TYPES,
            num_positions=cfg.MODEL.SEM_SEG_HEAD.NUM_POSITIONS,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
            losses=cfg.MODEL.MASK_FORMER.LOSSES,
            type_weight=cfg.MODEL.MASK_FORMER.TYPE_WEIGHT,
            position_weight=cfg.MODEL.MASK_FORMER.POSITION_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO
        )
        
        return base_config

    def prepare_targets(self, targets, images):
        # 扩展目标数据准备
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # 提取必要字段
            gt_boxes = targets_per_image.gt_boxes
            gt_classes = targets_per_image.gt_classes
            h, w = targets_per_image.image_size  # 原始图像尺寸
            
            device = gt_boxes.device  # 确保设备一致
            
            padded_masks = targets_per_image.gt_masks
            # polygons = targets_per_image.gt_masks.polygons  # 获取多边形列表
            
            # # 为每个多边形生成二进制掩码
            # bitmasks = [
            #     torch.from_numpy(polygons_to_bitmask(poly, h, w)).to(device=device)
            #     for poly in polygons
            # ]
            # gt_masks = torch.stack(bitmasks)
            # gt_masks = targets_per_image.gt_masks
            # padded_masks = torch.zeros(
            #     (gt_masks.shape[0], h_pad, w_pad),
            #     dtype=gt_masks.dtype, device=gt_masks.device
            # )
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
                "type_ids": targets_per_image.gt_type_ids,
                "position_ids": targets_per_image.gt_position_ids
            })
        return new_targets

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * "image": Tensor, image in (C, H, W) format.
                * "instances": per-region ground truth
                * Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model (may be different
                    from input resolution), used in inference.

        Returns:
            list[dict]: each dict has the results for one image.
                The dict contains the following keys:
                * "sem_seg": A Tensor representing per-pixel segmentation predicted by the head.
                * "panoptic_seg": A tuple representing panoptic output
                panoptic_seg (Tensor): (height, width) of ids for each segment.
                segments_info (list[dict]): Describes each segment in `panoptic_seg`.
        """
        # Process input images
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # Backbone feature extraction
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        
        # print("+===================+OPTS MODEL$$$$$$$$$$$$$$$$$$$$$")
        # for key, values in outputs.items():
        #     try:
        #         print(f"{key}: {values[-1][1]}")
        #     except KeyError: 
        #         print("okfine")

        if self.training:
            # During training, prepare targets and calculate losses
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # Calculate losses
            losses = self.criterion(outputs, targets)

            # Apply loss weights if specified
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)

            return losses
        else:
            # Non-training (inference) phase
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            
            mask_type_results = outputs["pred_types"]
            mask_position_results = outputs["pred_positions"]

            # Upsample masks to the original image size
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                # Postprocess if needed
                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # Perform semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # Perform panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # Perform instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

                # Add pred_types and pred_positions during inference (non-training)
                # Calculate pred_types and pred_positions from the decoder output
                # decoder_output = self.sem_seg_head.predictor.decoder_norm(
                #     self.sem_seg_head.predictor.query_feat.weight
                # )
                # processed_results[-1]["pred_types"] = self.sem_seg_head.predictor.type_embed(decoder_output)
                # processed_results[-1]["pred_positions"] = self.sem_seg_head.predictor.position_embed(decoder_output)
                
                processed_results[-1]["pred_types"] = mask_type_results[-1]
                processed_results[-1]["pred_positions"] = mask_position_results[-1]

            return processed_results
