import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraVisualizer:
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager

    def extract_positions(self):
        """提取相机中心位置（世界坐标系中的相机位置）"""
        positions = []
        directions = []
        for cam in self.camera_manager.cameras:
            extrinsic = cam['extr']
            R = extrinsic[:, :3]  # 旋转矩阵
            t = extrinsic[:, 3]   # 平移向量
            cam_center = -np.linalg.inv(R) @ t  # 计算相机中心

            # 计算相机方向（Z 轴方向，第三列）
            cam_direction = R[:, 2]  
            positions.append(cam_center)
            directions.append(cam_direction)
        return np.array(positions), np.array(directions)

    def visualize(self, show_interp=True):
        """可视化相机位置和方向"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        positions, directions = self.extract_positions()

        # 绘制相机位置
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o', label="Camera Positions")

        # 绘制相机朝向
        for i in range(len(positions)):
            ax.quiver(*positions[i], *directions[i], length=0.1, color='b')

        # 可视化插值相机位置
        if show_interp:
            interp_positions = []
            for r in np.linspace(0, 1, num=50):
                extr, _ = self.camera_manager.get_interpolated(r)
                R_interp = extr[:, :3]
                t_interp = extr[:, 3]
                interp_cam_center = -np.linalg.inv(R_interp) @ t_interp
                interp_positions.append(interp_cam_center)
            interp_positions = np.array(interp_positions)
            ax.plot(interp_positions[:, 0], interp_positions[:, 1], interp_positions[:, 2], 'g--', label="Interpolated Path")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title("Camera Positions and Interpolated Path")
        plt.show()
