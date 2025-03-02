from __future__ import print_function, division
import sys
import os

# 获取 LiveDemo 根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_ROOT)

# 添加 gps_gaussian 目录到 sys.path
GPS_GAUSSIAN_PATH = os.path.join(SRC_ROOT, "gpsgaussian")
if GPS_GAUSSIAN_PATH not in sys.path:
    sys.path.append(GPS_GAUSSIAN_PATH)

# 一系列的导包

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# 正文
class GPSRenderer:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

    def infer_frame(self, view_select, ratio=0.5):
        item = self.dataset.get_test_item(0, source_id=view_select)
        data = self.fetch_data(item)
        data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
        
        with torch.no_grad():
            data, _, _ = self.model(data, is_train=False)

        # 提取点云数据
        xyz_valid = []
        rgb_valid = []
        rot_valid = []
        scale_valid = []
        opacity_valid = []

        for view in ['lmain', 'rmain']:
            valid = data[view]['pts_valid'][0, :]
            xyz = data[view]['xyz'][0, :, :]
            rgb = data[view]['img'][0, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot = data[view]['rot_maps'][0, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale = data[view]['scale_maps'][0, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity = data[view]['opacity_maps'][0, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_valid.append(xyz[valid].view(-1, 3))
            rgb_valid.append(rgb[valid].view(-1, 3))
            rot_valid.append(rot[valid].view(-1, 4))
            scale_valid.append(scale[valid].view(-1, 3))
            opacity_valid.append(opacity[valid].view(-1, 1))

        pts_xyz = torch.cat(xyz_valid, dim=0).cpu().numpy()
        pts_rgb = (torch.cat(rgb_valid, dim=0).cpu().numpy())
        pts_rgb = (pts_rgb * 0.5 + 0.5) * 255
        pts_dc = np.ones((pts_xyz.shape[0], 3))
        pts_rot = torch.cat(rot_valid, dim=0).cpu().numpy()
        pts_scale = torch.cat(scale_valid, dim=0).cpu().numpy()
        pts_opacity = torch.cat(opacity_valid, dim=0).cpu().numpy()

        nx = np.ones((pts_xyz.shape[0], 1))
        ny = np.zeros((pts_xyz.shape[0], 1))
        nz = np.zeros((pts_xyz.shape[0], 1))
    
        # 保存为 PLY 格式
        ply_path = os.path.join(self.cfg.test_out_path, f'novel.ply')
        self.save_ply(pts_xyz, pts_rgb, pts_opacity, pts_scale, pts_rot, nx, ny, nz, pts_dc, ply_path)
    
    def save_ply(self, pts_xyz, pts_rgb, pts_opacity, pts_scale, pts_rot, nx, ny, nz, pts_dc, ply_path, background_color=(1.0, 1.0, 1.0)):
        assert isinstance(background_color, tuple) and len(background_color) == 3
        assert all(isinstance(c, float) for c in background_color)
    
        # 创建 PLY 文件头
        header = '''ply
format binary_little_endian 1.0
element vertex {0}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float red
property float green
property float blue
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
'''.format(pts_xyz.shape[0])
    
        # 将点云数据和颜色，即每个点的数据为 (x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, red, green, blue, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3)
        pts_data = np.hstack((pts_xyz, nx, ny, nz, pts_dc, pts_rgb, pts_opacity, pts_scale, pts_rot))
    
        # 保存为 PLY 文件
        with open(ply_path, 'wb') as f:
            f.write(header.encode('utf-8'))
            pts_data.astype(np.float32).tofile(f)

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")

def setup_renderer(test_data_root, ckpt_path, out_path, cfg_path, src_view, ratio=0.5):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg_for_train = os.path.join(cfg_path, 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = ckpt_path
    cfg.test_out_path = out_path
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = GPSRenderer(cfg, phase='test')
    return render

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()

    render = setup_renderer(args.test_data_root, args.ckpt_path, args.src_view, args.ratio)
    render.infer_frame(view_select=args.src_view, ratio=args.ratio)