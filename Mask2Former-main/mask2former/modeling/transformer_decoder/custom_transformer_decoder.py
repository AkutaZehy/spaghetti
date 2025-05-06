from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder, MLP

import torch
import torch.nn as nn
from torch.nn import functional as F

class MobileBottleneckClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, expansion=4):
        super().__init__()
        hidden_dim = input_dim * expansion
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.block(x)

@TRANSFORMER_DECODER_REGISTRY.register()
class CustomMultiScaleMaskedTransformerDecoder(MultiScaleMaskedTransformerDecoder):
    '''
    fc -> bottleneck
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.mask_classification:
            hidden_dim = self.query_feat.weight.shape[1]
            num_classes = self.class_embed.out_features - 1
            
            self.class_embed = MobileBottleneckClassifier(
                input_dim=hidden_dim,
                output_dim=num_classes + 1,  # 保持与原始维度一致
                expansion=4
            )

@TRANSFORMER_DECODER_REGISTRY.register()
class FCX3(MultiScaleMaskedTransformerDecoder):
    '''
    fc -> 3xfc
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.mask_classification:
            hidden_dim = self.query_feat.weight.shape[1]
            num_classes = self.class_embed.out_features - 1
            
            hidden_dim_2 = hidden_dim
            hidden_dim_3 = hidden_dim
            
            self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
            
@TRANSFORMER_DECODER_REGISTRY.register()
class FCBE(MultiScaleMaskedTransformerDecoder):
    '''
    mlp -> fc/einsum
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mask_classification:
            # 确保class_embed能处理mask_embed的输出维度
            self.class_embed = nn.Linear(self.mask_embed.layers[-1].out_features, self.class_embed.out_features)

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        
        mask_embed = self.mask_embed(decoder_output)
        outputs_class = self.class_embed(mask_embed)
        
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
    
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
class QA(MultiScaleMaskedTransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.mask_classification:
            hidden_dim = self.query_feat.weight.shape[1]
            num_classes = self.class_embed.out_features - 1
            
            # 替换为新的注意力分类头
            self.class_embed = QueryAwareClassifier(
                input_dim=hidden_dim,
                output_dim=num_classes + 1,
                expansion=2  # 温和扩展
            )

