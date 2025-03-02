# tools/frames2video.py
import os
import cv2
from natsort import natsorted

def convert_frames_to_video(input_dir, output_path, fps=30):
    frames = natsorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    if not frames:
        return
    
    # 读取第一帧获取尺寸
    sample = cv2.imread(os.path.join(input_dir, frames[0]))
    h, w, _ = sample.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame_name in tqdm(frames):
        frame = cv2.imread(os.path.join(input_dir, frame_name))
        writer.write(frame)
    writer.release()

# 批量转换所有机位（假设原始数据按 cam_00, cam_01... 组织）
for cam_id in range(16):
    input_dir = f"原始数据路径/cam_{cam_id:02d}"
    output_path = f"data/input_videos/cam_{cam_id:02d}.mp4"
    convert_frames_to_video(input_dir, output_path)