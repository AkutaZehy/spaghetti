import copy
import torch

from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
from detectron2.structures import Instances, PolygonMasks

class MultiTaskMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        # 原始处理流程
        dd = copy.deepcopy(dataset_dict)
        
        dataset_dict = super().__call__(dataset_dict)
        
        # 添加额外字段
        instances = dataset_dict["instances"]
        annotations = dd.get("annotations", [])
        
        if not hasattr(instances, "gt_masks"):
            segms = [ann["segmentation"] for ann in annotations]
            instances.gt_masks = PolygonMasks(segms)
        
        # 添加type_id和position_id
        instances.gt_type_ids = torch.tensor(
            [ann["type_id"] for ann in annotations], dtype=torch.int64
        )
        instances.gt_position_ids = torch.tensor(
            [ann["position_id"] for ann in annotations], dtype=torch.int64
        )
        
        return dataset_dict