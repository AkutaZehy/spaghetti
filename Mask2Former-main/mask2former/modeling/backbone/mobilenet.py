import torch
import torch.nn as nn
import torchvision.models as models
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from .mobilenetv3 import MobileNetV3
from detectron2.layers import ShapeSpec

@BACKBONE_REGISTRY.register()
class D2MobileNetV3(MobileNetV3, Backbone):
    def __init__(self, cfg, input_shape):
        # 获取配置参数
        arch_type = cfg.MODEL.MOBILENET.ARCHITECTURE
        width_mult = cfg.MODEL.MOBILENET.WIDTH_MULT
        pretrained = cfg.MODEL.MOBILENET.PRETRAINED
        
        # 初始化MobileNetV3
        super().__init__(mode=arch_type, width_mult=width_mult)
        
        # 配置输出特征
        self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
        
        # 根据模型类型设置特征步长和通道数
        if arch_type == "large":
            self._out_feature_strides = {
                "res2": 4,
                "res3": 8,
                "res4": 16,
                "res5": 32,
            }
            self._out_feature_channels = {
                "res2": int(16 * width_mult),
                "res3": int(24 * width_mult),
                "res4": int(40 * width_mult),
                "res5": int(112 * width_mult),
            }
        else:  # small
            self._out_feature_strides = {
                "res2": 4,
                "res3": 8,
                "res4": 16,
                "res5": 32,
            }
            self._out_feature_channels = {
                "res2": int(16 * width_mult),
                "res3": int(24 * width_mult),
                "res4": int(48 * width_mult),
                "res5": int(96 * width_mult),
            }

        # 加载预训练权重
        if pretrained:
            self._load_pretrained(arch_type, width_mult)

    def _load_pretrained(self, arch_type, width_mult):
        # 这里可以添加预训练权重加载逻辑
        pass

    def forward(self, x):
        """
        前向传播，返回指定特征层的输出
        """
        features = {}
        x = self.features[0](x)  # 第一层卷积
        x = self.features[1](x)  # BN + Hswish
        features["res2"] = x
        
        # 遍历后续层并提取特征
        for i in range(2, len(self.features)):
            x = self.features[i](x)
            if i == 3:  # res3
                features["res3"] = x
            elif i == 6:  # res4
                features["res4"] = x
            elif i == 12:  # res5
                features["res5"] = x
        
        return {k: v for k, v in features.items() if k in self._out_features}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

# @BACKBONE_REGISTRY.register()
# class D2PretrainedMobileNetV3(Backbone):
#     def __init__(self, cfg, input_shape):
#         super().__init__()
#         self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
#         self.width_mult = cfg.MODEL.MOBILENET.WIDTH_MULT

#         # 加载 PyTorch 的预训练 MobileNetV3-Large 模型
#         self.mobilenet = models.mobilenet_v3_large(pretrained=False)  # width_mult 固定为 1.0（PyTorch 官方模型不支持动态 width_mult）

#         # 截断 features 层，移除最后一层（输出 960 的层）
#         self.features = nn.Sequential(*list(self.mobilenet.features.children())[:-1])

#         # 添加转换层将 160 通道 → 112 * width_mult
#         out_channels = int(112 * self.width_mult)  # 关键修正
#         self.res5_conv = nn.Conv2d(160, out_channels, kernel_size=1, bias=False)
#         self.res5_bn = nn.BatchNorm2d(out_channels)
#         self.res5_act = nn.Hardswish(inplace=True)

#         # 定义特征层的通道数和 stride
#         self._out_feature_channels = {
#             "res2": int(16 * self.width_mult),
#             "res3": int(24 * self.width_mult),
#             "res4": int(40 * self.width_mult),
#             "res5": int(112 * self.width_mult),
#         }
#         self._out_feature_strides = {
#             "res2": 4,
#             "res3": 8,
#             "res4": 16,
#             "res5": 32,
#         }

#     def forward(self, x):
#         features = {}
#         for i, layer in enumerate(self.features):
#             x = layer(x)
#             if i == 1:   # res2（对应层索引 1）
#                 features["res2"] = x
#             elif i == 3: # res3（对应层索引 3）
#                 features["res3"] = x
#             elif i == 6: # res4（对应层索引 6）
#                 features["res4"] = x
#             elif i == len(self.features) - 1:  # 最后一层（输出 160）
#                 pass  # 等待转换层处理

#         # 处理 res5 的转换
#         x = self.res5_conv(x)
#         x = self.res5_bn(x)
#         x = self.res5_act(x)
#         features["res5"] = x

#         return {k: v for k, v in features.items() if k in self._out_features}

#     def output_shape(self):
#         return {
#             name: ShapeSpec(
#                 channels=self._out_feature_channels[name],
#                 stride=self._out_feature_strides[name]
#             )
#             for name in self._out_features
#         }

#     @property
#     def size_divisibility(self):
#         return 32

@BACKBONE_REGISTRY.register()
class D2PretrainedMobileNetV3(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self._out_features = cfg.MODEL.MOBILENET.OUT_FEATURES

        # 加载预训练的 MobileNetV3-Large 模型
        self.mobilenet = models.mobilenet_v3_large(pretrained=True)
        # self.mobilenet = models.mobilenet_v3_large(pretrained=False)

        # 截断模型，只保留到倒数第二层（不包括最后的分类层）
        self.features = nn.Sequential(*list(self.mobilenet.features.children()))

        # 定义特征层的通道数和步长
        self._out_feature_channels = {
            "res2": 24,
            "res3": 40,
            "res4": 80,
            "res5": 160,
        }
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C, H, W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"MobileNetV3 takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 3:  # res2
                outputs["res2"] = x
            elif i == 6:  # res3
                outputs["res3"] = x
            elif i == 10:  # res4
                outputs["res4"] = x
            elif i == 15:  # res5
                outputs["res5"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32