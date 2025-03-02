import numpy as np

class CameraManager:
    def __init__(self, param_dir):
        self.cameras = []
        for cam_id in range(16):
            extr = np.load(f"{param_dir}/{cam_id}_extrinsic.npy")  # 3x4 矩阵
            intr = np.load(f"{param_dir}/{cam_id}_intrinsic.npy")  # 3x3 矩阵
            self.cameras.append({'extr': extr, 'intr': intr})

        # 环形相机连接关系
        self.adjacency = [(i, (i+1) % 16) for i in range(16)]

    def _lerp_extrinsics(self, extr1, extr2, ratio):
        """线性插值外参矩阵"""
        R1, t1 = extr1[:, :3], extr1[:, 3]
        R2, t2 = extr2[:, :3], extr2[:, 3]

        # 线性插值平移
        t_interp = (1 - ratio) * t1 + ratio * t2

        # 线性插值旋转（LERP），然后正交化
        R_interp = (1 - ratio) * R1 + ratio * R2
        U, _, Vt = np.linalg.svd(R_interp)  # SVD 使其正交化
        R_interp = U @ Vt

        return np.hstack([R_interp, t_interp.reshape(-1, 1)])

    def get_interpolated(self, ratio):
        """获取插值后的相机外参"""
        idx = int(ratio * 16) % 16
        next_idx = (idx + 1) % 16
        local_ratio = ratio * 16 - idx  # 局部插值比例 [0,1)

        return self._lerp_extrinsics(
            self.cameras[idx]['extr'],
            self.cameras[next_idx]['extr'],
            local_ratio
        ), self.cameras[idx]['intr']
