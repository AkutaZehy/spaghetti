import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json as original_load_coco_json
from detectron2.structures import BoxMode

# TEST
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "datasets"))
os.environ["DETECTRON2_DATASETS"] = DATA

def custom_load_coco_json(json_file, image_root, dataset_name=None):
    """
    自定义加载函数：保留 COCO JSON 中所有的 annotation 字段，
    包括额外字段 "type_id" 和 "position_id"。
    """
    with open(json_file, "r") as f:
        coco = json.load(f)

    # 构造 image id -> image 信息的字典
    imgs = {img["id"]: img for img in coco["images"]}
    # 构造每个图像的记录，初始化 annotations 列表
    imgid_to_record = {
        img_id: {
            "file_name": os.path.join(image_root, img_info["file_name"]),
            "height": img_info["height"],
            "width": img_info["width"],
            "image_id": img_id,
            "annotations": []
        }
        for img_id, img_info in imgs.items()
    }

    # 遍历所有 annotation，并将所有字段全部复制进来
    for ann in coco["annotations"]:
        # 复制整个 annotation 字典，确保保留所有字段
        ann_record = {k: v for k, v in ann.items()}
        
        # 将 category_id, type_id, position_id 减去 1
        ann_record["category_id"] -= 1
        # ann_record["type_id"] -= 1
        # ann_record["position_id"] -= 1
        
        # 设置标准的 bbox_mode（这里假设 bbox 为 [x, y, w, h]）
        ann_record["bbox_mode"] = BoxMode.XYWH_ABS
        # 把这个 annotation 加入对应的图像记录
        imgid_to_record[ann["image_id"]]["annotations"].append(ann_record)

    # 返回所有图像记录的列表
    return list(imgid_to_record.values())
  
def register_defectt(root):
    """
    注册自定义缺陷数据集，同时保留额外的 annotation 字段。
    数据集目录结构：
      root/
         defectp/
           annotations/
             instances_train2025.json
             instances_val2025.json
           train2025/       # 训练图片目录
           val2025/         # 验证图片目录
    """
    defect_root = os.path.join(root, "defectp")
    dataset_splits = [
        ("defectp_train2025", "annotations/instances_train2025.json", "train2025"),
        ("defectp_val2025", "annotations/instances_val2025.json", "val2025"),
    ]
    thing_classes = ["surface", "edge"]
    
    for dname, ann_file, img_dir in dataset_splits:
        ann_path = os.path.join(defect_root, ann_file)
        img_path = os.path.join(defect_root, img_dir)
        DatasetCatalog.register(
            dname,
            lambda ann_path=ann_path, img_path=img_path: custom_load_coco_json(ann_path, img_path, dname)
        )
        MetadataCatalog.get(dname).set(
            thing_classes=thing_classes,
            evaluator_type="coco"
        )
        print(f"Registered dataset: {dname}")

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_defectt(_root)