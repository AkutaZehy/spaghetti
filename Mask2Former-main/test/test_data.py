import os, sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, VisImage, GenericMask
from mask2former import data

class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0):
        super().__init__(img_rgb, metadata, scale)
        self.img_rgb = img_rgb

    def draw_dataset_dict(self, d):
        img = self.img_rgb.copy()
        mask_layer = np.zeros_like(img, dtype=np.uint8)  # 用于遮罩绘制

        if "annotations" in d:
            for ann in d["annotations"]:
                bbox = ann["bbox"]
                x, y, w, h = bbox

                # 处理偏移问题：将 category_id 从 1 开始调整为从 0 开始
                cat_id = ann["category_id"]
                
                # print(self.metadata)
                
                try:
                    cat_name = self.metadata.thing_classes[cat_id] if cat_id is not None else "bg"
                except Exception:
                    cat_name = "UNK"

                # 处理 type_id 和 position_id 偏移
                type_id = ann.get("type_id", None)
                pos_id = ann.get("position_id", None)
                try:
                    type_name = self.metadata.type_classes[type_id] if type_id is not None else ""
                except Exception:
                    type_name = ""
                try:
                    pos_name = self.metadata.position_classes[pos_id] if pos_id is not None else ""
                except Exception:
                    pos_name = ""

                # 构建标签字符串
                label_str = cat_name
                if type_name:
                    label_str += f", {type_name}"
                if pos_name:
                    label_str += f", {pos_name}"

                # 绘制 segmentation 遮罩
                if "segmentation" in ann:
                    try:
                        mask_obj = GenericMask(ann["segmentation"], img.shape[0], img.shape[1])
                        mask = mask_obj.mask.astype(np.uint8)  # (H, W) bool -> uint8

                        # color = [int(255 * random.random()) for _ in range(3)]  # 随机颜色
                        color = [255 for _ in range(3)]
                        for c in range(3):
                            mask_layer[:, :, c] = np.where(mask == 1, color[c], mask_layer[:, :, c])
                    except Exception as e:
                        print("Segmentation mask draw error:", e)

                # 绘制边框
                cv2.rectangle(
                    img,
                    (int(x), int(y)),
                    (int(x + w), int(y + h)),
                    (0, 255, 0),
                    2,
                )

                # 绘制文本
                cv2.putText(
                    img,
                    label_str,
                    (int(x) - 50, int(y) - 10),
                    # (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 合并遮罩图层和原图
        # blended = cv2.addWeighted(img, 1.0, mask_layer, 0.5, 0)
        
        blended = mask_layer

        return VisImage(blended)


def test_dataset(dataset_name):
    # 从 DatasetCatalog 获取数据集
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print("Dataset {} loaded. Total {} images.".format(dataset_name, len(dataset_dicts)))
    # 打印第一条数据的部分信息
    # print("First record:", dataset_dicts[0])
    
    # 随机展示一张图片及其标注
    d = random.choice(dataset_dicts)
    img = cv2.imread(d["file_name"])
    if img is None:
        print("无法读取图片：", d["file_name"])
        return

    # Convert BGR to RGB for matplotlib display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    visualizer = CustomVisualizer(img, metadata=metadata, scale=0.5)
    vis_output = visualizer.draw_dataset_dict(d)
    plt.figure(figsize=(12, 8))
    plt.imshow(vis_output.get_image())
    plt.axis("off")
    plt.title(d["file_name"])
    plt.savefig('check.png', dpi=300)
    plt.show()

def save_all_images(dataset_name, output_folder):
    # 从 DatasetCatalog 获取数据集
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    print("Dataset {} loaded. Total {} images.".format(dataset_name, len(dataset_dicts)))
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历数据集中的每一张图片
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        if img is None:
            print("无法读取图片：", d["file_name"])
            continue

        # Convert BGR to RGB for matplotlib display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualizer = CustomVisualizer(img, metadata=metadata, scale=0.5)
        vis_output = visualizer.draw_dataset_dict(d)

        # 保存可视化结果
        # output_file = os.path.join(output_folder, os.path.basename(d["file_name"]).replace('.bmp', '.png'))
        # plt.figure(figsize=(12, 8))
        # plt.imshow(vis_output.get_image())
        # plt.axis("off")
        # plt.title(d["file_name"])
        # plt.savefig(output_file, dpi=300)
        # plt.close()
        
        # Convert back to BGR for cv2.imwrite
        vis_output_img = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

        # 保存可视化结果
        output_file = os.path.join(output_folder, os.path.basename(d["file_name"]).replace('.bmp', '.png'))
        cv2.imwrite(output_file, vis_output_img)


if __name__ == "__main__":
    # 测试注册的训练数据集
    # test_dataset("defect_train2025")
    
    # 也可以测试验证数据集
    # test_dataset("defect_val2025")
    
    # save_all_images("defect_val2025", "mask")
    save_all_images("defect_train2025", "mask")
