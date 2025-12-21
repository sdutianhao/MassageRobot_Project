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

        # 1) Pose 初始化（可训练）
        if gt_pose_np is not None:
            pose_init = torch.from_numpy(gt_pose_np).float().clone()

            # 维度对齐
            if pose_init.shape[1] != 46:
                pose_temp = torch.zeros((1, 46)).float()
                valid = min(pose_init.shape[1], 46)
                pose_temp[:, :valid] = pose_init[:, :valid]
                pose_init = pose_temp

            # 只对 theta 加噪（符合 Stage2 设想）
            if init_noise_std > 0:
                print(f"[SkelAdapter] Injecting Micro-Perturbation (Std={init_noise_std} * 5.0)...")
                noise = torch.randn_like(pose_init) * (init_noise_std * 5.0)
                pose_init += noise
        else:
            pose_init = torch.zeros((1, 46)).float()

        # 2) Beta 初始化（固定，不训练）
        if gt_beta_np is not None:
            beta_init = torch.from_numpy(gt_beta_np).float().clone()
        else:
            beta_init = torch.zeros((1, 10)).float()

        self.pose = nn.Parameter(pose_init)
        self.register_buffer("beta", beta_init)
        self.register_buffer("trans", torch.zeros((1, 3)).float())
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
