import torch
import torch.nn as nn

from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from detectron2.config import configurable
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiTaskTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(
        self,
        cfg,  # 新增配置参数
        in_channels: int,
        mask_classification: bool,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        num_types: int,  # 多任务参数
        num_positions: int,  # 多任务参数
        **kwargs
    ):
        # 使用父类的完整属性
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            **kwargs
        )
        # 多任务预测头
        self.type_embed = nn.Linear(hidden_dim, num_types + 1)
        self.position_embed = nn.Linear(hidden_dim, num_positions + 1)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # 保持父类逻辑并扩展多任务输出
        outputs_class, outputs_mask, attn_mask = super().forward_prediction_heads(
            output, mask_features, attn_mask_target_size
        )
        
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        outputs_type = self.type_embed(decoder_output)
        outputs_position = self.position_embed(decoder_output)

        return {
            "pred_logits": outputs_class,
            "pred_masks": outputs_mask,
            "attn_mask": attn_mask,
            "pred_types": outputs_type,
            "pred_positions": outputs_position
        }

    def forward(self, x, mask_features, mask=None):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_type = []
        predictions_position = []

        # ⛳ 使用子类版本的 forward_prediction_heads
        pred_dict = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(pred_dict["pred_logits"])
        predictions_mask.append(pred_dict["pred_masks"])
        predictions_type.append(pred_dict["pred_types"])
        predictions_position.append(pred_dict["pred_positions"])
        attn_mask = pred_dict["attn_mask"]

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)

            pred_dict = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(pred_dict["pred_logits"])
            predictions_mask.append(pred_dict["pred_masks"])
            predictions_type.append(pred_dict["pred_types"])
            predictions_position.append(pred_dict["pred_positions"])
            attn_mask = pred_dict["attn_mask"]

        return {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_types': predictions_type[-1],
            'pred_positions': predictions_position[-1],
            'aux_outputs': self._set_aux_loss_multitask(
                predictions_class, predictions_mask, predictions_type, predictions_position
            )
        }

    def _set_aux_loss_multitask(self, cls_preds, mask_preds, type_preds, position_preds):
        return [
            {
                "pred_logits": a,
                "pred_masks": b,
                "pred_types": c,
                "pred_positions": d
            }
            for a, b, c, d in zip(cls_preds[:-1], mask_preds[:-1], type_preds[:-1], position_preds[:-1])
        ]

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        # 获取父类配置
        base_config = super().from_config(cfg, in_channels, mask_classification)
        # 添加多任务参数
        base_config.update({
            "cfg": cfg,
            "num_types": cfg.MODEL.SEM_SEG_HEAD.NUM_TYPES,
            "num_positions": cfg.MODEL.SEM_SEG_HEAD.NUM_POSITIONS
        })
        return base_config
    
class QueryAwareClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, expansion=2, groups=4):
        super().__init__()
        # 保持原初始化参数不变
        self.hidden_dim = int(input_dim * expansion)
        
        self.group_fc = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # 修改SE模块结构
        self.se = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),  # 直接处理通道维度
            nn.ReLU(),
            nn.Linear(self.hidden_dim//4, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # x: [B, Q, C]
        x = self.group_fc(x)  # [B, Q, H]
        
        # 通道注意力计算修正
        channel_avg = x.mean(dim=1, keepdim=True)  # [B, 1, H]
        se_weight = self.se(channel_avg)           # [B, 1, H]
        x = x * se_weight
        
        return self.final_fc(x)

@TRANSFORMER_DECODER_REGISTRY.register()
class QAMultiTaskTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    @configurable
    def __init__(
        self,
        cfg,  # 新增配置参数
        in_channels: int,
        mask_classification: bool,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        num_types: int,  # 多任务参数
        num_positions: int,  # 多任务参数
        **kwargs
    ):
        # 使用父类的完整属性
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            **kwargs
        )
        # 多任务预测头-QA
        if self.mask_classification:
            hidden_dim = self.query_feat.weight.shape[1]
            num_classes = self.class_embed.out_features - 1
            
            # 替换为新的注意力分类头
            self.class_embed = QueryAwareClassifier(
                input_dim=hidden_dim,
                output_dim=num_classes + 1,
                expansion=2  # 温和扩展
            )
            
        self.type_embed = QueryAwareClassifier(
                input_dim=hidden_dim,
                output_dim=num_types + 1,
                expansion=2  # 温和扩展
            )
        self.position_embed = QueryAwareClassifier(
                input_dim=hidden_dim,
                output_dim=num_positions + 1,
                expansion=2  # 温和扩展
            )

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # 保持父类逻辑并扩展多任务输出
        outputs_class, outputs_mask, attn_mask = super().forward_prediction_heads(
            output, mask_features, attn_mask_target_size
        )
        
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        outputs_type = self.type_embed(decoder_output)
        outputs_position = self.position_embed(decoder_output)

        return {
            "pred_logits": outputs_class,
            "pred_masks": outputs_mask,
            "attn_mask": attn_mask,
            "pred_types": outputs_type,
            "pred_positions": outputs_position
        }

    def forward(self, x, mask_features, mask=None):
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_type = []
        predictions_position = []

        # ⛳ 使用子类版本的 forward_prediction_heads
        pred_dict = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(pred_dict["pred_logits"])
        predictions_mask.append(pred_dict["pred_masks"])
        predictions_type.append(pred_dict["pred_types"])
        predictions_position.append(pred_dict["pred_positions"])
        attn_mask = pred_dict["attn_mask"]

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )
            output = self.transformer_ffn_layers[i](output)

            pred_dict = self.forward_prediction_heads(
                output, mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(pred_dict["pred_logits"])
            predictions_mask.append(pred_dict["pred_masks"])
            predictions_type.append(pred_dict["pred_types"])
            predictions_position.append(pred_dict["pred_positions"])
            attn_mask = pred_dict["attn_mask"]

        return {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_types': predictions_type[-1],
            'pred_positions': predictions_position[-1],
            'aux_outputs': self._set_aux_loss_multitask(
                predictions_class, predictions_mask, predictions_type, predictions_position
            )
        }

    def _set_aux_loss_multitask(self, cls_preds, mask_preds, type_preds, position_preds):
        return [
            {
                "pred_logits": a,
                "pred_masks": b,
                "pred_types": c,
                "pred_positions": d
            }
            for a, b, c, d in zip(cls_preds[:-1], mask_preds[:-1], type_preds[:-1], position_preds[:-1])
        ]

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        # 获取父类配置
        base_config = super().from_config(cfg, in_channels, mask_classification)
        # 添加多任务参数
        base_config.update({
            "cfg": cfg,
            "num_types": cfg.MODEL.SEM_SEG_HEAD.NUM_TYPES,
            "num_positions": cfg.MODEL.SEM_SEG_HEAD.NUM_POSITIONS
        })
        return base_config