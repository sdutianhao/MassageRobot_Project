import sys
import os
import torch
import torch.nn as nn
import numpy as np

SKEL_PATH = "/home/hsmr/MassageRobot_Project/HSMR4Robot/thirdparty/SKEL"
if SKEL_PATH not in sys.path:
    sys.path.append(SKEL_PATH)

from skel.skel_model import SKEL

class SkelAdapter(nn.Module):
    def __init__(self, gender='female', gt_pose_np=None, gt_beta_np=None, init_noise_std=0.0, device=None):
        super().__init__()
        
        MODEL_DIR = "/home/hsmr/MassageRobot_Project/HSMR4Robot/data_inputs/body_models/skel"
        self.skel_model = SKEL(gender=gender, model_path=MODEL_DIR)
        
        # 1. Pose 初始化 (使用 clone 防止污染真值)
        if gt_pose_np is not None:
            pose_init = torch.from_numpy(gt_pose_np).float().clone()
            
            # 维度对齐
            if pose_init.shape[1] != 46:
                pose_temp = torch.zeros((1, 46)).float()
                valid = min(pose_init.shape[1], 46)
                pose_temp[:, :valid] = pose_init[:, :valid]
                pose_init = pose_temp
            
            # [随机微扰] 恢复高斯噪声
            # init_noise_std=0.02 * 5.0 = 0.1 rad (约 5.7度)
            # 这个幅度足以产生肉眼可见的错位，但不会破坏骨骼拓扑
            if init_noise_std > 0:
                print(f"[SkelAdapter] Injecting Micro-Perturbation (Std={init_noise_std} * 5.0)...")
                noise = torch.randn_like(pose_init) * (init_noise_std * 5.0) 
                pose_init += noise
        else:
            pose_init = torch.zeros((1, 46)).float()

        # 2. Beta 初始化 (使用 clone)
        if gt_beta_np is not None:
            beta_init = torch.from_numpy(gt_beta_np).float().clone()
            
            # [随机微扰] 轻微改变胖瘦
            if init_noise_std > 0:
                noise_beta = torch.randn_like(beta_init) * (init_noise_std * 10.0)
                beta_init += noise_beta
        else:
            beta_init = torch.zeros((1, 10)).float()
            
        self.pose = nn.Parameter(pose_init)
        self.beta = nn.Parameter(beta_init)
        self.trans = nn.Parameter(torch.zeros((1, 3)).float()) # 默认在地面
        self.faces = self.skel_model.skin_f.cpu().numpy()

    def forward_vertices(self):
        output = self.skel_model(
            poses=self.pose, 
            betas=self.beta, 
            trans=self.trans,
            poses_type='skel',
            skelmesh=False
        )
        return output.skin_verts[0]
