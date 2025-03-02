import sys
sys.path.append('..')

import os
import shutil
import cv2
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import open3d as o3d

# 导入各子模块（请确保路径正确）
from livecore.camera_utils import CameraManager
from livecore.matting_engine import MattingWrapper
from livecore.gps2renderer import setup_renderer

# ---------------------------
# 工具函数：目录检查
def check_dir(path_list, clear=True):
    for path in path_list:
        if clear:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        else:
            if not os.path.exists(path):
                os.makedirs(path)

# ---------------------------
# 全局配置
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
video_dir = os.path.join(root, 'data/input_videos')
data_dir = os.path.join(root, 'data/cache/real_data')
frame_dir = os.path.join(data_dir, 'img/0')
mask_dir = os.path.join(data_dir, 'mask/0')
output_dir = os.path.join(root, 'data/cache/output')

# matting_model_dir = os.path.join(root, 'models/rvm_mobilenetv3.pth')
matting_model_dir = os.path.join(root, 'models/rvm_resnet50.pth')
gps_model_dir = os.path.join(root, 'models/GPS-GS_stage2_final.pth')
cfg_dir = os.path.join(root, 'configs/gaussian_config')

default_src_view_index = 0  # 取值 0~15，形成对 [i, (i+1)%16]
ratio = 0.5

check_dir([frame_dir, mask_dir, output_dir])

# ---------------------------
# 初始化 Matting 模块
mw = MattingWrapper(matting_model_dir)

def process_video(video_name, frame_id):
    """
    处理单个视频：从指定帧提取图像，保存原始帧和通过 matting 生成的 mask
    """
    video_path = os.path.join(video_dir, f'{video_name}.mp4')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return False
    # 保存原始帧
    frame_path = os.path.join(frame_dir, f'{video_name}.jpg')
    cv2.imwrite(frame_path, frame)
    # 生成 mask（matting 处理）
    matting_frame = mw.process_frame(frame)
    matting_frame = cv2.cvtColor(matting_frame, cv2.COLOR_GRAY2RGB)
    mask_path = os.path.join(mask_dir, f'{video_name}.png')
    cv2.imwrite(mask_path, (matting_frame * 255).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cap.release()
    return True

def init_renderer(src_view):
    """
    初始化 renderer，src_view 为一个成对的视角参数
    """
    return setup_renderer(data_dir, gps_model_dir, output_dir, cfg_dir, src_view, ratio)

def get_src_view_from_index(index):
    """
    根据视角起始索引生成视角对，例如 index=0 -> [0,1]，index=15 -> [15,0]
    """
    return [index, (index + 1) % 16]

# ---------------------------
# Open3D 查看器线程：持续创建窗口并周期性更新显示的点云
class Open3DViewerThread(QtCore.QThread):
    def __init__(self, initial_ply_path, parent=None):
        super().__init__(parent)
        self.initial_ply_path = initial_ply_path
        self.new_ply_path = None
        self.mutex = QtCore.QMutex()
        self.vis = None

    def run(self):
        # 创建并初始化 Open3D Visualizer（非阻塞更新方式）
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="PLY Viewer", width=800, height=600)
        # 初始加载：如果文件存在且有效，则加载
        if os.path.exists(self.initial_ply_path) and os.path.getsize(self.initial_ply_path) > 0:
            try:
                pcd = o3d.io.read_point_cloud(self.initial_ply_path)
                self.vis.add_geometry(pcd)
            except Exception as e:
                print("Initial load geometry failed:", e)
        # 持续更新循环
        while not self.isInterruptionRequested():
            self.mutex.lock()
            if self.new_ply_path is not None:
                try:
                    new_pcd = o3d.io.read_point_cloud(self.new_ply_path)
                    if new_pcd.has_points():
                        self.vis.clear_geometries()
                        self.vis.add_geometry(new_pcd)
                        print("Open3D Viewer 已更新新几何体。")
                except Exception as e:
                    print("Exception updating geometry:", e)
                self.new_ply_path = None
            self.mutex.unlock()
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.03)
        self.vis.destroy_window()

    @QtCore.pyqtSlot(str)
    def update_geometry(self, ply_path):
        self.mutex.lock()
        self.new_ply_path = ply_path
        self.mutex.unlock()

# ---------------------------
# PyQt5 主窗口，集成步进渲染和视角控制
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LIVE DEMO")
        self.current_frame = 0
        self.total_frames = self.get_total_frames()
        self.step_value = 5  # 默认步长
        self.current_view_index = default_src_view_index
        self.src_view = get_src_view_from_index(self.current_view_index)
        self.renderer = init_renderer(self.src_view)
        self.setup_ui()
        # 启动 Open3D Viewer 线程，窗口独立且持续运行，不阻塞 PyQt
        self.open3d_viewer_thread = Open3DViewerThread(os.path.join(output_dir, "novel.ply"))
        self.open3d_viewer_thread.start()
        
    def setup_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("复位")
        self.btn_reset.clicked.connect(self.reset_frame)
        controls_layout.addWidget(self.btn_reset)
        
        self.lbl_frame = QtWidgets.QLabel(f"当前帧: {self.current_frame}")
        controls_layout.addWidget(self.lbl_frame)
        
        controls_layout.addWidget(QtWidgets.QLabel("步长:"))
        self.step_input = QtWidgets.QLineEdit(str(self.step_value))
        self.step_input.setFixedWidth(50)
        controls_layout.addWidget(self.step_input)
        
        self.btn_step = QtWidgets.QPushButton("步进渲染")
        self.btn_step.clicked.connect(self.step_render)
        controls_layout.addWidget(self.btn_step)
        
        self.btn_view_left = QtWidgets.QPushButton("视角左")
        self.btn_view_left.clicked.connect(self.view_left)
        controls_layout.addWidget(self.btn_view_left)
        
        self.btn_view_right = QtWidgets.QPushButton("视角右")
        self.btn_view_right.clicked.connect(self.view_right)
        controls_layout.addWidget(self.btn_view_right)
        
        layout.addLayout(controls_layout)
        
    def get_total_frames(self):
        sample_video = os.path.join(video_dir, "0.mp4")
        cap = cv2.VideoCapture(sample_video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total

    def process_current_frame(self):
        print(f"处理当前帧 {self.current_frame} ...")
        success = True
        for i in range(16):
            video_name = f"{i}"
            if not process_video(video_name, self.current_frame):
                success = False
                break
        return success

    def render_current_frame(self):
        # 调用 renderer 生成 ply 文件（生成 novel.ply）
        self.renderer.infer_frame(view_select=self.src_view, ratio=ratio)
        print("调用 renderer.infer_frame 完成，novel.ply 应已生成。")
        # 更新 Open3D 查看器的显示（非阻塞更新）
        ply_path = os.path.join(output_dir, "novel.ply")
        self.open3d_viewer_thread.update_geometry(ply_path)

    def step_render(self):
        try:
            self.step_value = int(self.step_input.text())
        except ValueError:
            self.step_value = 5
            self.step_input.setText("5")
        if self.process_current_frame():
            self.render_current_frame()
            new_frame = self.current_frame + self.step_value
            if new_frame >= self.total_frames:
                new_frame = new_frame % self.total_frames
            self.current_frame = new_frame
            self.lbl_frame.setText(f"当前帧: {self.current_frame}")
    
    def reset_frame(self):
        self.current_frame = 0
        self.lbl_frame.setText(f"当前帧: {self.current_frame}")
        
    def view_left(self):
        self.current_view_index = (self.current_view_index - 1) % 16
        self.src_view = get_src_view_from_index(self.current_view_index)
        self.renderer = init_renderer(self.src_view)
        if self.process_current_frame():
            self.render_current_frame()
    
    def view_right(self):
        self.current_view_index = (self.current_view_index + 1) % 16
        self.src_view = get_src_view_from_index(self.current_view_index)
        self.renderer = init_renderer(self.src_view)
        if self.process_current_frame():
            self.render_current_frame()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    exit_code = app.exec_()
    # 程序退出时终止 Open3D 线程
    window.open3d_viewer_thread.requestInterruption()
    window.open3d_viewer_thread.wait()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
