# main_window.py
import os
import sys
import shutil
import threading
import cv2
import time
import numpy as np
import torch
import math
from tqdm import tqdm
from dearpygui import dearpygui as dpg

# 获取 LiveDemo 根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_ROOT)

# 添加 gps_gaussian 目录到 sys.path
GPS_GAUSSIAN_PATH = os.path.join(SRC_ROOT, "gpsgaussian")
if GPS_GAUSSIAN_PATH not in sys.path:
    sys.path.append(GPS_GAUSSIAN_PATH)

# 导入各子模块（确保路径正确）
from livecore.camera_utils import CameraManager
from livecore.matting_engine import MattingWrapper
from livecore.gps_renderer import setup_renderer

from gaussian_renderer import render
from lib.graphics_utils import getWorld2View2

# ---------------------------
# 工具函数：目录检查
def check_dir(path_list, clear=True):
    # 返回值是原先的文件夹是否为空
    isEmpty = True
    # 如果 path_list 仅有一个元素，将其转换为列表
    if not isinstance(path_list, list):
        path_list = [path_list]
    for path in path_list:
        if clear:
            if os.path.exists(path):
                shutil.rmtree(path)
                isEmpty = False
            os.makedirs(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                isEmpty = False
    return isEmpty

# ---------------------------
# 全局配置
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
video_dir = os.path.join(root, 'data/input_videos')
data_dir = os.path.join(root, 'data/cache/real_data')
# frame_dir = os.path.join(data_dir, 'img/0')
frame_dir = os.path.join(data_dir, 'img')
# mask_dir = os.path.join(data_dir, 'mask/0')
mask_dir = os.path.join(data_dir, 'mask')
config_parm_dir = os.path.join(root, "configs/camera_params")
parm_dir = os.path.join(data_dir, 'parm')
output_dir = os.path.join(root, 'data/cache/output')
font_dir = os.path.join(root, 'data/font', 'SourceHanSansCN-Regular.otf')

# 模型路径（此处可根据需要选择不同模型）
matting_model_dir = os.path.join(root, 'models/rvm_mobilenetv3.pth')
# matting_model_dir = os.path.join(root, 'models/rvm_resnet50.pth')
gps_model_dir = os.path.join(root, 'models/GPS-GS_stage2_final.pth')
cfg_dir = os.path.join(root, 'configs/gaussian_config')

cache_clear = False
# cache_clear = True

# is_resize = False
is_resize = True

check_dir([frame_dir, mask_dir, output_dir, parm_dir], clear=cache_clear)

# ---------------------------
# 初始化 Matting 模块
mw = MattingWrapper(matting_model_dir)

def process_video(video_name, frame_id):
    """
    对单个视频：取指定帧，保存原始帧与 mask
    """
    video_path = os.path.join(video_dir, f'{video_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    # 保存原始帧（缓存用，可选）
    # frame_path = os.path.join(frame_dir, f'{video_name}.jpg')
    frame_dir_id = os.path.join(frame_dir, f'{frame_id}')
    check_dir(frame_dir_id, clear=False) # 保留缓存
    frame_path = os.path.join(frame_dir_id, f'{video_name}.jpg')
    cv2.imwrite(frame_path, frame)

    # 生成 mask
    matting_frame = mw.process_frame(frame)
    # 注意 OpenCV 默认 BGR -> 需要转换为 RGB
    matting_frame = cv2.cvtColor(matting_frame, cv2.COLOR_GRAY2RGB)
    mask_dir_id = os.path.join(mask_dir, f'{frame_id}')
    check_dir(mask_dir_id, clear=False) # 保留缓存
    mask_path = os.path.join(mask_dir_id, f'{video_name}.png')
    cv2.imwrite(mask_path, (matting_frame * 255).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cap.release()

    # 生成相机参数，将 config_parm_dir 中的相机参数复制到 parms_dir 的对应目录下
    camera_dir = os.path.join(data_dir, 'parm', f'{frame_id}')
    check_dir(camera_dir, clear=False) # 保留缓存
    for file in os.listdir(config_parm_dir):
        shutil.copy(os.path.join(config_parm_dir, file), os.path.join(camera_dir, file))
    return True

video_frames = []

print("预载可能需要一段时间，请耐心等待 ...")

def load_videos_to_memory(video_dir):
    start_time = time.time()
    # print("开始加载所有视频到内存 ...")
    # print("当前时间: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    for i in tqdm(range(16)):  # 假设总共有16个视频
        video_name = f"{i}.mp4"
        video_path = os.path.join(video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_id in range(total_frames):  # 使用 for 循环读取每一帧
            ret, frame = cap.read()
            if not ret:  # 如果读取失败（理论上不会发生，但保留以防万一）
                print(f"在帧 {frame_id} 处读取失败")
                break
            if is_resize:
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
            frames.append(frame)  # 将帧添加到列表中
        cap.release()
        video_frames.append(frames)  # 将每个视频的所有帧存储到全局变量中
    # print("所有视频已加载到内存！")
    # print("耗时: ", time.time() - start_time)

load_videos_to_memory(video_dir)
# print("视频帧数: ", len(video_frames[0]))

img_list = []
# mask_list = []

def video2cache(frame_id):
    """
    1. 清空 img_list, mask_list
    2. 取定帧，更新 img_list, mask_list
    """
    global img_list
    
    img_list.clear()
    # mask_list.clear()
    for i in range(16):
        # video_name = f"{i}"
        # video_path = os.path.join(video_dir, f'{video_name}.mp4')
        # cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # ret, frame = cap.read()
        # if not ret:
            # cap.release()
            # return False
        if frame_id >= len(video_frames[i]):  # 检查帧号是否超出范围
            print(f"视频 {i} 的帧数不足，无法读取第 {frame_id} 帧！")
            return False
        
        frame = video_frames[i][frame_id]  # 从内存中读取指定帧
        if is_resize:
            frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        img_list.append(frame)
        # matting_frame = mw.process_frame(frame)
        # matting_frame = cv2.cvtColor(matting_frame, cv2.COLOR_GRAY2RGB) * 255
        # mask_list.append(matting_frame)
    # mask_list = mw.process_frames(img_list)
        # cap.release()
    # print("img_list, mask_list 更新完成 ... current_frame: ", frame_id)
    return True

def init_renderer(src_view, intr_list, extr_list):
    """
    初始化 GPSRenderer（原 gps_renderer.py 的 setup_renderer）
    """
    return setup_renderer(data_dir, gps_model_dir, output_dir, cfg_dir, src_view, ratio, intr_list, extr_list)

def get_src_view_from_index(index):
    """
    根据视角索引生成视角对，例如 0 -> [0,1], 15 -> [15,0]
    """
    return [index, (index + 1) % 16]

# 全局变量
current_frame = 0
step = 1
progress_value = 0.0
total_frames = 450
camera_count = 16
angle = 0

intr_list = []
extr_list = []

positions = [0, 0, 0]
rotations = [0, 0, 0]

for i in range(camera_count):
    intr_name = os.path.join(config_parm_dir, f'{i}_intrinsic.npy')
    extr_name = os.path.join(config_parm_dir, f'{i}_extrinsic.npy')
    intr_list.append(np.load(intr_name))
    extr_list.append(np.load(extr_name))

# 检查缓存
def check_cache(frame_id):
    frame_folder = os.path.join(frame_dir, f'{frame_id}')
    check_dir(frame_folder, False)
    count = len(os.listdir(frame_folder))
    return count == camera_count

# --------------------------- 
# GUI 部分

print("正在启动 GUI ...")
start_frame_dir = os.path.join(frame_dir, f'{current_frame}')
# if not check_cache(current_frame):
#     print("首次加载可能较慢，请耐心等待 ...")
#     for i in tqdm(range(16)):
#             video_name = f"{i}"
#             process_video(video_name, 0)
src_view = get_src_view_from_index(0)
ratio = 0.5
renderer = init_renderer(src_view, intr_list, extr_list)
# current_img = renderer.infer_frame(view_select=src_view, ratio=ratio)

mask_group = []

def img2mask():
    # 仅生成 src_view 视角的 mask
    global mask_group, img_list, src_view
    start_time = time.time() # 计时
    mask_group.clear()
    img0, img1 = img_list[src_view[0]], img_list[src_view[1]]
    mask0, mask1 = mw.process_frames([img0, img1])
    mask_group = [mask0, mask1]
    # print("耗时: ", time.time() - start_time)

video2cache(current_frame)
img2mask()

# current_img = renderer.infer_frame_from_cache(view_select=src_view, img_list=img_list, mask_list=mask_list, ratio=ratio, frame_id=current_frame)
current_img = renderer.infer_frame_from_cache(view_select=src_view, img_list=img_list, mask_list=mask_group, ratio=ratio, frame_id=current_frame, position=positions, rotation=rotations)

frame_width = 1600
frame_height = 900
box = 400

def img_to_texture(img):
    width, height, channels = img.shape
    # 将 img 转换为 RGBA 格式
    img_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    img_rgba[:, :, :3] = img
    img_rgba[:, :, 3] = 255
    # 交换R和B通道
    img_rgba = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGRA)
    data = img_rgba.flatten()/255.0

    return width, height, channels, data

# ---------------------------
# 主函数入口
dpg.create_context()

# 加载字体
with dpg.font_registry():
    with dpg.font(font_dir, 15) as myfont:
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Simplified_Common)
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
    dpg.bind_font(myfont)

dpg.create_viewport(title='Renderer Live Demo', width=frame_width, height=frame_height)

current_camera = os.path.join(frame_dir, '%d/%d.jpg')
camera_1 = 0
camera_2 = 4

width, height, channels, data = img_to_texture(current_img)
# width_1, height_1, channels_1, data_1 = dpg.load_image(current_camera % (current_frame, camera_1))
# width_2, height_2, channels_2, data_2 = dpg.load_image(current_camera % (current_frame, camera_2))
width_1, height_1, channels_1, data_1 = img_to_texture(img_list[camera_1])
width_2, height_2, channels_2, data_2 = img_to_texture(img_list[camera_2])

# width, height = box*2, box*2
# width_1, height_1 = box, box
# width_2, height_2 = box, box

with dpg.texture_registry():
    dpg.add_dynamic_texture(width, height, data, tag="image_id")
    dpg.add_dynamic_texture(width_1, height_1, data_1, tag="camera_id_1")
    dpg.add_dynamic_texture(width_2, height_2, data_2, tag="camera_id_2")

with dpg.window(tag='渲染图'):
    dpg.draw_image("camera_id_1", [0, 0], [box, box])
    dpg.draw_image("camera_id_2", [0, box], [box, box*2])
    dpg.draw_image("image_id", [box, 0], [box*3, box*2])
        
with dpg.window(label="位置控制帮助", modal=True, tag="position_help", no_title_bar=True, no_close=True, width=400, height=200, pos=[600, 200]):
    dpg.add_text("X轴代表屏幕的左右方向（默认值：0），向右为正。")
    dpg.add_text("Y轴代表屏幕的上下方向（默认值：0），向上为正。")
    dpg.add_text("Z轴代表屏幕的前后方向（默认值：0），向前为正。")
    dpg.add_button(label="好", width=200, pos=[100, 150], callback=lambda: dpg.configure_item("position_help", show=False))
    
with dpg.window(label="旋转控制帮助", modal=True, tag="rotation_help", no_title_bar=True, no_close=True, width=400, height=200, pos=[600, 200]):
    dpg.add_text("rX代表沿着屏幕的水平轴旋转（默认值：0），顺时针旋转为正。")
    dpg.add_text("rY代表沿着屏幕的垂直轴旋转（默认值：0），顺时针旋转为正。")
    dpg.add_text("rZ代表垂直于屏幕的轴旋转（默认值：0），顺时针旋转为正。")
    dpg.add_button(label="好", width=200, pos=[100, 150], callback=lambda: dpg.configure_item("rotation_help", show=False))

with dpg.window(label="控制面板", pos=[1250, 0], tag="控制面板", width=350, height=900, no_close=True, no_move=True):

    # 配置一个进度条，用于显示跳转进度
    dpg.add_text("渲染进度")
    dpg.add_progress_bar(label="进度", default_value=0.0, tag="progress_frame")

    dpg.add_spacing(count=10)

    dpg.add_text(f"当前帧: {current_frame}", tag="text_current_frame")

    with dpg.group(horizontal=True):
        dpg.add_button(label="跳转到", callback=lambda: jump_to_frame(dpg.get_value("input_frame")))
        dpg.add_text("第")
        dpg.add_input_int(label="帧", tag="input_frame", width=-50, default_value=current_frame)

    # 添加播放与暂停按钮
    dpg.add_text("播放控制")
    dpg.add_text("设置播放步长")
    dpg.add_same_line()
    dpg.add_input_int(label="帧", default_value=step, tag="input_step", width=-50, callback=lambda s,a,u: step_render_callback())
    with dpg.group(horizontal=True):
        dpg.add_button(label="播放", callback=lambda: start_playback())
        dpg.add_button(label="暂停", callback=lambda: stop_playback())

    dpg.add_spacing(count=10)

    # 设置观察机位
    dpg.add_text(f'设置视频机位')
    with dpg.group(horizontal=True):
        dpg.add_text("第一机位")
        dpg.add_input_int(default_value=camera_1, tag="camera_1", width=-150, callback=lambda s,a,u :set_view1(dpg.get_value("camera_1")))
    with dpg.group(horizontal=True):
        dpg.add_text("第二机位")
        dpg.add_input_int(default_value=camera_2, tag="camera_2", width=-150, callback=lambda s,a,u: set_view2(dpg.get_value("camera_2")))

    dpg.add_spacing(count=10)

    # 添加一个旋转角度的滑块，比例保留一位小数
    dpg.add_text("渲染控制")

    dpg.add_text(f"当前左机位：{src_view[0]}, 右机位：{src_view[1]}", tag="text_view")
    with dpg.group(horizontal=True):
        dpg.add_text("机位旋转")
        dpg.add_slider_float(default_value=0.0, min_value=-180.0, max_value=180.0, tag="slider_angle", callback=lambda s,a,u: set_view(dpg.get_value("slider_angle")))
    
    dpg.add_spacing(count=10)

    # 配置世界坐标位移与旋转的修改
    with dpg.group(horizontal=True):
        dpg.add_text("世界坐标位移")
        dpg.add_button(label="?", callback=lambda: dpg.configure_item("position_help", show=True))
    with dpg.group(horizontal=True):
        dpg.add_text("X")
        dpg.add_slider_float(default_value=0.0, min_value=-2.0, max_value=2.0, tag="slider_x", callback=lambda s,a,u: set_position(dpg.get_value("slider_x"), positions[1], positions[2]))
        dpg.add_text("m")
    with dpg.group(horizontal=True):
        dpg.add_text("Y")
        dpg.add_slider_float(default_value=0.0, min_value=-2.0, max_value=2.0, tag="slider_y", callback=lambda s,a,u: set_position(positions[0], dpg.get_value("slider_y"), positions[2]))
        dpg.add_text("m")
    with dpg.group(horizontal=True):
        dpg.add_text("Z")
        dpg.add_slider_float(default_value=0.0, min_value=-2.0, max_value=2.0, tag="slider_z", callback=lambda s,a,u: set_position(positions[0], positions[1], dpg.get_value("slider_z")))
        dpg.add_text("m")
    dpg.add_spacing(count=10)
    
    with dpg.group(horizontal=True):
        dpg.add_text("世界坐标旋转")
        dpg.add_button(label="?", callback=lambda: dpg.configure_item("rotation_help", show=True))
    with dpg.group(horizontal=True):
        dpg.add_text("rX")
        dpg.add_slider_float(default_value=0.0, min_value=-180.0, max_value=180.0, tag="slider_rx", callback=lambda s,a,u: set_rotation(dpg.get_value("slider_rx"), rotations[1], rotations[2]))
        dpg.add_text("°")
    with dpg.group(horizontal=True):
        dpg.add_text("rY")
        dpg.add_slider_float(default_value=0.0, min_value=-180.0, max_value=180.0, tag="slider_ry", callback=lambda s,a,u: set_rotation(rotations[0], dpg.get_value("slider_ry"), rotations[2]))
        dpg.add_text("°")
    with dpg.group(horizontal=True):
        dpg.add_text("rZ")
        dpg.add_slider_float(default_value=0.0, min_value=-180.0, max_value=180.0, tag="slider_rz", callback=lambda s,a,u: set_rotation(rotations[0], rotations[1], dpg.get_value("slider_rz")))
        dpg.add_text("°")
        
    dpg.add_spacing(count=10)
    
    dpg.add_button(label="重置变换", callback=lambda: reset())
        
# 渲染回调
def render_callback():
    global current_frame, renderer, src_view, ratio, current_camera, camera_1, camera_2, positions, rotations
    if camera_1 >= 16 or camera_2 >= 16 or camera_1 < 0 or camera_2 < 0:
        return
    if not video2cache(current_frame):
        return
    # current_img_new = renderer.infer_frame(view_select=src_view, ratio=ratio, frame_id=current_frame)
    img2mask()
    # current_img_new = renderer.infer_frame_from_cache(view_select=src_view, img_list=img_list, mask_list=mask_list, ratio=ratio, frame_id=current_frame)
    current_img_new = renderer.infer_frame_from_cache(view_select=src_view, img_list=img_list, mask_list=mask_group, ratio=ratio, frame_id=current_frame, position=positions, rotation=rotations)
    # width_1, height_1, channels_1, data_1 = dpg.load_image(current_camera % (current_frame, camera_1))
    # width_2, height_2, channels_2, data_2 = dpg.load_image(current_camera % (current_frame, camera_2))
    width, height, channels, data = img_to_texture(current_img_new)
    width_1, height_1, channels_1, data_1 = img_to_texture(img_list[camera_1])
    width_2, height_2, channels_2, data_2 = img_to_texture(img_list[camera_2])
    dpg.set_value("image_id", data)
    dpg.set_value("camera_id_1", data_1)
    dpg.set_value("camera_id_2", data_2)
    dpg.set_value("text_current_frame", f"当前帧: {current_frame}")

def jump_to_frame(data):
    global current_frame
    print(f"跳转到帧 {data}")
    current_frame = int(data)
    # render_cache(current_frame)
    render_callback()

# def render_cache(frame_id):
#     global progress_value
#     success = True
#     frame_dir_id = os.path.join(frame_dir, f'{frame_id}')
#     mask_dir_id = os.path.join(mask_dir, f'{frame_id}')
#     if check_cache(frame_id): return True
#     for i in tqdm(range(16)):
#         video_name = f"{i}"
#         if not process_video(video_name, frame_id):
#             success = False
#             break
#         progress_value = (i + 1) / 16
#         dpg.set_value("progress_frame", progress_value)
#     if not success:
#         return False
#     print(f"已处理当前帧 {frame_id} ...")
#     return True

def set_view(value):
    # 将 -180 ~ 180 角度映射到 src_view 的 0 ~ 15 以及 ratio 的 0 ~ 1
    global src_view, ratio, angle
    # 其中 angle 为 0 ~ 360 的角度，angle = 0 时 src_view = [0, 1], ratio = 0.5
    # angle = 180 时 src_view = [8, 9], ratio = 0.5
    # angle = -180 时 src_view = [8, 9], ratio = 0.5
    angle = value % 360
    index = int(angle / 22.5) % 16
    src_view = get_src_view_from_index(index)
    ratio = (angle + 11.25) % 22.5 / 22.5
    dpg.set_value("text_view", f"当前左机位：{src_view[0]}, 右机位：{src_view[1]}")
    render_callback()

def set_view1(value):
    global camera_1, camera_2
    if -1 < value and value < 16:
        if value != camera_2:
            camera_1 = value
    render_callback()

def set_view2(value):
    global camera_1, camera_2
    if -1 < value and value < 16:
        if value!= camera_1:
            camera_2 = value
    render_callback()

def set_position(x, y, z):
    global positions
    positions = [x, y*-1, z*-1]
    render_callback()
    
def set_rotation(rx, ry, rz):
    global rotations
    rotations = [rx, ry*-1, rz]
    render_callback()
    
def reset():
    global positions, rotations, camera_1, camera_2
    positions = [0, 0, 0]
    rotations = [0, 0, 0]
    camera_1 = 0
    camera_2 = 4
    
    dpg.set_value("slider_x", 0.0)
    dpg.set_value("slider_y", 0.0)
    dpg.set_value("slider_z", 0.0)
    dpg.set_value("slider_rx", 0.0)
    dpg.set_value("slider_ry", 0.0)
    dpg.set_value("slider_rz", 0.0)
    dpg.set_value("slider_angle", 0.0)
    
    set_view(0)

def step_render_callback():
    global step
    try:
        step = int(dpg.get_value("input_step"))
    except ValueError:
        step = 1
        dpg.set_value("input_step", 1)

playback_active = False

# 监听播放状态，每隔一段时间更新帧
def playback_listener(gap=0.5):
    global playback_active, step, current_frame, total_frames
    while playback_active:
        current_frame += step
        if current_frame > total_frames:
            current_frame = 0
        # render_cache(current_frame)
        render_callback()
        time.sleep(gap)

def start_playback():
    global playback_active
    playback_active = True

    # 启动播放线程
    playback_thread = threading.Thread(target=playback_listener)
    playback_thread.start()

def stop_playback():
    global playback_active
    playback_active = False

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("渲染图", True)
dpg.start_dearpygui()
dpg.destroy_context()
