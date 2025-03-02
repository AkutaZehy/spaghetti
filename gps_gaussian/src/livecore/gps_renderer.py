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

from lib.human_loader import StereoHumanDataset, stereo_pts2flow
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GPSRenderer:
    def __init__(self, cfg_file, phase, intr_list=None, extr_list=None):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.intr_list = intr_list
        self.extr_list = extr_list

        self.opt = self.cfg.dataset

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.opt, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

    def infer_frame(self, view_select, ratio=0.5, frame_id=0):
        # 取测试数据中的第一项
        item = self.dataset.get_test_item(frame_id, source_id=view_select)
        data = self.fetch_data(item)
        data = get_novel_calib(data, self.opt, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
        
        with torch.no_grad():
            data, _, _ = self.model(data, is_train=False)
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

        render_novel = self.tensor2np(data['novel_view']['img_pred'])

        return render_novel
    
    def infer_frame_from_cache(self, view_select, img_list, mask_list, ratio=0.5, frame_id=0):
        item = self.get_dict(img_list, mask_list, view_select, frame_id)
        data = self.fetch_data(item)
        data = get_novel_calib(data, self.opt, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
        
        with torch.no_grad():
            data, _, _ = self.model(data, is_train=False)
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

        render_novel = self.tensor2np(data['novel_view']['img_pred'])

        return render_novel

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

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np
    
    def load_single_view(self, img_list, mask, source_id):
        # frame_list, mask_list, source_id -> frame, mask, intr, extr
        # return img_list[source_id], mask_list[source_id], self.intr_list[source_id], self.extr_list[source_id], None
        return img_list[source_id], mask, self.intr_list[source_id], self.extr_list[source_id], None
        
    def get_dict(self, img_list, mask_list, source, frame_id):
        # eg. source=[0,1]
        # img_list, mask_list, source_id -> dict_tensor
        view0_data = self.load_single_view(img_list, mask_list[0], source[0])
        view1_data = self.load_single_view(img_list, mask_list[1], source[1])
        lmain_intr_ori, lmain_extr_ori = view0_data[2], view0_data[3]
        rmain_intr_ori, rmain_extr_ori = view1_data[2], view1_data[3]
        stereo_np = self.get_rectified_stereo_data(main_view_data=view0_data, ref_view_data=view1_data)
        dict_tensor = self.stereo_to_dict_tensor(stereo_np, frame_id)

        dict_tensor['lmain']['intr_ori'] = torch.FloatTensor(lmain_intr_ori)
        dict_tensor['rmain']['intr_ori'] = torch.FloatTensor(rmain_intr_ori)
        dict_tensor['lmain']['extr_ori'] = torch.FloatTensor(lmain_extr_ori)
        dict_tensor['rmain']['extr_ori'] = torch.FloatTensor(rmain_extr_ori)

        img_len = self.opt.src_res * 2 if self.opt.use_hr_img else self.opt.src_res
        novel_dict = {
            'height': torch.IntTensor([img_len]),
            'width': torch.IntTensor([img_len])
        }

        dict_tensor.update({
            'novel_view': novel_dict
        })

        return dict_tensor

    def get_rectified_stereo_data(self, main_view_data, ref_view_data):
        img0, mask0, intr0, extr0, pts0 = main_view_data
        img1, mask1, intr1, extr1, pts1 = ref_view_data

        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB) # BGR -> RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # BGR -> RGB

        H, W = self.opt.src_res, self.opt.src_res
        r0, t0 = extr0[:3, :3], extr0[:3, 3:]
        r1, t1 = extr1[:3, :3], extr1[:3, 3:]
        inv_r0 = r0.T
        inv_t0 = - r0.T @ t0
        E0 = np.eye(4)
        E0[:3, :3], E0[:3, 3:] = inv_r0, inv_t0
        E1 = np.eye(4)
        E1[:3, :3], E1[:3, 3:] = r1, t1
        E = E1 @ E0
        R, T = E[:3, :3], E[:3, 3]
        dist0, dist1 = np.zeros(4), np.zeros(4)

        R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(intr0, dist0, intr1, dist1, (W, H), R, T, flags=0)

        new_extr0 = R0 @ extr0
        new_intr0 = P0[:3, :3]
        new_extr1 = R1 @ extr1
        new_intr1 = P1[:3, :3]
        Tf_x = np.array(P1[0, 3])

        camera = {
            'intr0': new_intr0,
            'intr1': new_intr1,
            'extr0': new_extr0,
            'extr1': new_extr1,
            'Tf_x': Tf_x
        }

        rectify_mat0_x, rectify_mat0_y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
        new_img0 = cv2.remap(img0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        new_mask0 = cv2.remap(mask0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
        rectify_mat1_x, rectify_mat1_y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)
        new_img1 = cv2.remap(img1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        new_mask1 = cv2.remap(mask1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)
        rectify0 = new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y
        rectify1 = new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y

        stereo_data = {
            'img0': new_img0,
            'mask0': new_mask0,
            'img1': new_img1,
            'mask1': new_mask1,
            'camera': camera
        }

        if pts0 is not None:
            flow0, flow1 = stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x)

            kernel = np.ones((3, 3), dtype=np.uint8)
            flow_eroded, valid_eroded = [], []
            for (flow, new_mask) in [(flow0, new_mask0), (flow1, new_mask1)]:
                valid = (new_mask.copy()[:, :, 0] / 255.0).astype(np.float32)
                valid = cv2.erode(valid, kernel, 1)
                valid[valid >= 0.66] = 1.0
                valid[valid < 0.66] = 0.0
                flow *= valid
                valid *= 255.0
                flow_eroded.append(flow)
                valid_eroded.append(valid)

            stereo_data.update({
                'flow0': flow_eroded[0],
                'valid0': valid_eroded[0].astype(np.uint8),
                'flow1': flow_eroded[1],
                'valid1': valid_eroded[1].astype(np.uint8)
            })

        return stereo_data
    
    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'Tf_x': torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'Tf_x': -torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        if 'flow0' in stereo_data:
            flow_tensor, valid_tensor = [], []
            for (flow_view, valid_view) in [('flow0', 'valid0'), ('flow1', 'valid1')]:
                flow = torch.from_numpy(stereo_data[flow_view])
                flow = torch.unsqueeze(flow, dim=0)
                flow_tensor.append(flow)

                valid = torch.from_numpy(stereo_data[valid_view])
                valid = torch.unsqueeze(valid, dim=0)
                valid = valid / 255.0
                valid_tensor.append(valid)

            lmain_data['flow'], lmain_data['valid'] = flow_tensor[0], valid_tensor[0]
            rmain_data['flow'], rmain_data['valid'] = flow_tensor[1], valid_tensor[1]

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}
    
def setup_renderer(test_data_root, ckpt_path, out_path, cfg_path, src_view, ratio=0.5, intr_list=None, extr_list=None):
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

    render_obj = GPSRenderer(cfg, phase='test', intr_list=intr_list, extr_list=extr_list)
    return render_obj

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    args = parser.parse_args()

    renderer = setup_renderer(args.test_data_root, args.ckpt_path, args.src_view, args.ratio)
    renderer.infer_frame(view_select=args.src_view, ratio=args.ratio)
