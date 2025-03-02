import torch
from torchvision.transforms import ToTensor

import sys
sys.path.append('..')
from matting.model import MattingNetwork

import cv2

class MattingWrapper:
    def __init__(self, model_path):
        self.model = MattingNetwork().eval().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.rec = [None] * 4  # 保持记忆状态
        self.downsample_ratio = 0.25  # 与模型匹配

    def process_frame(self, frame_bgr):
        """输入 OpenCV BGR 帧，输出 alpha matte"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = ToTensor()(frame_rgb).unsqueeze(0).cuda()
        
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(
                tensor, *self.rec, 
                downsample_ratio=self.downsample_ratio
            )
        
        return pha[0].cpu().numpy().squeeze()

    def process_frames(self, frames_bgr):
        """输入 OpenCV BGR 帧列表，输出 alpha matte 列表"""
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        tensors = torch.stack([ToTensor()(frame).unsqueeze(0) for frame in frames_rgb]).cuda()
        
        results = []
        with torch.no_grad():
            for tensor in tensors:
                fgr, pha, *self.rec = self.model(tensor, *self.rec, downsample_ratio=self.downsample_ratio)
                pha = pha[0].cpu().numpy().squeeze()
                pha = cv2.cvtColor(pha, cv2.COLOR_GRAY2RGB) * 255
                results.append(pha)
        
        return results