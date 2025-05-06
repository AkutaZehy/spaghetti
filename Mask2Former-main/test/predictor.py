# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import random

import cv2
import torch

import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, VisImage, GenericMask

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
                    type_name = "undefined"
                try:
                    pos_name = self.metadata.position_classes[pos_id] if pos_id is not None else ""
                except Exception:
                    pos_name = "undefined"

                # print(cat_id, type_id, pos_id, "=CTPiid in Cus")
                # print(cat_name, type_name, pos_name, "=CTP in Cus")

                # 构建标签字符串
                label_str = cat_name
                if type_name:
                    label_str += f", {type_name}"
                if pos_name:
                    label_str += f", {pos_name}"

                # 绘制 segmentation 遮罩
                if "segmentation" in ann:
                    try:
                        mask = np.array(ann["segmentation"], dtype=np.uint8)
                        
                        if mask.shape[:2] != img.shape[:2]:
                            print(f"Warning: Mask size {mask.shape[:2]} does not match image size {img.shape[:2]}. Resizing mask.")
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                        color = [int(255 * random.random()) for _ in range(3)]  # 随机颜色
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
                    # (int(x) - 50, int(y) - 10),
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # 合并遮罩图层和原图
        blended = cv2.addWeighted(img, 1.0, mask_layer, 0.5, 0)

        return VisImage(blended)

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        
        # print("PREDICTOR------------+-+-+-+-+-+-+-")
        # for key, values in predictions.items():
        #     print(f"{key}: {values[:3]}")
        
        image_rgb = image[:, :, ::-1]
        visualizer = CustomVisualizer(image_rgb, self.metadata, scale=1.0)

        # 检查 pred_types 和 pred_positions 是否存在
        pred_types = predictions.get("pred_types")
        pred_positions = predictions.get("pred_positions")

        # print("+=========================================+")

        class_categories = ["crack", "dent", "erosion", "scratch"]
        # class_categories = ["linear", "block", "point"]
        # class_categories = ["surface", "edge"]
        type_categories = ["linear", "block", "point"]
        position_categories = ["surface", "edge"]

        # 创建一个模拟的 dataset_dict
        dataset_dict = {
            "file_name": "predicted_image",
            "annotations": [],
            "metadata": self.metadata  # 添加 metadata
        }

        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            for i in range(len(instances)):
                logit = instances.scores[i]
                
                # if logit < 0.05: continue
                # if logit < 0.78: continue
                if logit > 0.1: continue
                # print(logit, "LO")
                
                # 提取每个实例的掩码、类别、位置和类型
                mask = instances.pred_masks[i]
                class_id = instances.pred_classes[i].item()

                # 检查 pred_types 和 pred_positions 是否为空
                if pred_types is not None:
                    type_id_list = pred_types[i]
                    # print(type_id_list, "LST?")
                    # tv, _ = torch.max(type_id_list[3], dim=0)
                    # if tv > 1: continue
                    # else: print(tv)
                    type_max_value, type_id = torch.max(type_id_list[:3], dim=0)
                else:
                    type_id = None  # 如果 pred_types 为空，设置为 None

                if pred_positions is not None:
                    position_id_list = pred_positions[i]
                    # pv, _ = torch.max(position_id_list[2], dim=0)
                    # if pv > 1: continue
                    position_max_value, position_id = torch.max(position_id_list[:2], dim=0)
                else:
                    position_id = None  # 如果 pred_positions 为空，设置为 None

                # print(type_id, position_id, "=TP")
                # print(type_max_value, position_max_value, "=TP max")
                # 如果 pred_types 或 pred_positions 为空，跳过相关处理
                if type_id is None or position_id is None:
                    predicted_type = None
                    predicted_position = None
                # elif type_id.item() == 3 or position_id.item() == 2: continue
                else:
                    # if type_max_value < 0.5 or position_max_value < 0.5:
                    if type_max_value < 2 or position_max_value > 3:
                    # if type_max_value < 0 or position_max_value < 0:
                        continue  # 如果置信度低于阈值，跳过这个实例
                    # print(type_id, position_id, "=TP alive")
                    print(type_max_value, position_max_value, "=Alive value")

                    try:
                        predicted_type = type_categories[type_id.item()-1]
                        predicted_position = position_categories[position_id.item()-1]
                    except IndexError: continue
                    # except IndexError:
                    #     predicted_type = type_categories[0]
                    #     predicted_position = position_categories[0]

                predicted_class = class_categories[class_id] if class_id < 4 else "undefined"

                # 计算掩码的边界框
                mask_bool = mask > 0.5  # 假设阈值为 0.5

                # 计算掩码的边界框
                y_indices, x_indices = np.where(mask_bool)
                if y_indices.size == 0 or x_indices.size == 0:
                    continue  # 如果没有非零像素，跳过这个实例

                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                # 创建一个模拟的 annotation
                annotation = {
                    "bbox": bbox,
                    "category_id": class_id,
                    "type_id": type_id.item() if type_id is not None else None,
                    "position_id": position_id.item() if position_id is not None else None,
                    "segmentation": mask.numpy().tolist()  # 将掩码转换为列表
                }
                
                # print(class_id, " ", type_id.item(), " ", position_id.item(), "=CTP")

                dataset_dict["annotations"].append(annotation)

        # 使用 CustomVisualizer 的 draw_dataset_dict 方法绘制
        vis_output = visualizer.draw_dataset_dict(dataset_dict)

        return predictions, vis_output

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
