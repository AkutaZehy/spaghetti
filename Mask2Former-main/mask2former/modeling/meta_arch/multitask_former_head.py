from typing import Callable, Dict, List, Optional, Tuple, Union

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from .mask_former_head import MaskFormerHead
from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..transformer_decoder.multitask_transformer_decoder import MultiTaskTransformerDecoder

@SEM_SEG_HEADS_REGISTRY.register()
class MultiTaskMaskFormerHead(MaskFormerHead):
    @configurable
    def __init__(
        self,
        *,
        num_types: int,
        num_positions: int,
        **kwargs
    ):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        base_config = super().from_config(cfg, input_shape)
        # 添加多任务参数
        base_config.update({
            "num_types": cfg.MODEL.SEM_SEG_HEAD.NUM_TYPES,
            "num_positions": cfg.MODEL.SEM_SEG_HEAD.NUM_POSITIONS
        })
        return base_config