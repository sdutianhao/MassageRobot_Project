import torch
import numpy as np
import os
import sys

# 引入项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage2.skel_adapter import SkelAdapter
from stage2.io_stage1 import load_stage1_result

class MockMeshGenerator:
    def __init__(self, device='cpu'):
        self.device = device

    def generate_data(self, stage1_npz, gt_npz_path, noise_std=0.015):
        """
        1. 读取 SKEL 真值参数 -> 生成 V_gt (6322 verts)
        2. 读取 Stage 1 相机参数 -> 投影顶点 -> 找到 ROI
        3. 对 ROI 顶点加噪 -> 生成 V_init
        """
        print(f"[Mock] Loading GT Params: {gt_npz_path}")
        gt_data = np.load(gt_npz_path, allow_pickle=True)
        gt_pose = gt_data['poses']
        gt_beta = gt_data['betas']

        # 1. 使用 SkelAdapter 生成完美的 GT 顶点 (init_noise_std=0.0 表示不加 theta 噪音)
        # 这里完全复用 Stage 2 的类
        adapter = SkelAdapter(
            gender='female',
            gt_pose_np=gt_pose,
            gt_beta_np=gt_beta,
            init_noise_std=0.0, 
            device=self.device
        ).to(self.device)
        
        # 获取真值顶点 (Blue Mesh)
        v_gt = adapter.forward_vertices().detach()
        faces = torch.tensor(adapter.faces.astype(np.int32), device=self.device).long()
        
        # 2. 计算 ROI 区域 (复用 Stage 1 结果)
        rot, trans, K, roi_xywh = load_stage1_result(stage1_npz, device=self.device)
        
        # 投影 3D -> 2D 以确定哪些顶点在 ROI 里
        # P_cam = V @ R.T + t
        v_cam = torch.matmul(v_gt, rot.transpose(0, 1)) + trans
        v_img_homo = torch.matmul(v_cam, K.transpose(0, 1))
        v_u = v_img_homo[:, 0] / v_img_homo[:, 2]
        v_v = v_img_homo[:, 1] / v_img_homo[:, 2]
        
        # ROI 范围
        rx, ry, rw, rh = roi_xywh
        mask_u = (v_u >= rx) & (v_u <= (rx + rw))
        mask_v = (v_v >= ry) & (v_v <= (ry + rh))
        roi_mask = mask_u & mask_v # 这里的 True 表示该顶点在 ROI 内
        
        print(f"[Mock] Total Verts: {len(v_gt)}, Verts in ROI: {roi_mask.sum()}")
        
        # 3. 生成带噪初始值 (Green Mesh)
        v_init = v_gt.clone()
        noise = torch.randn_like(v_gt) * noise_std
        
        # 只在 ROI 区域叠加噪声
        v_init[roi_mask] += noise[roi_mask]
        
        return v_gt, v_init, faces, (rot, trans, K, roi_xywh)

if __name__ == "__main__":
    pass
