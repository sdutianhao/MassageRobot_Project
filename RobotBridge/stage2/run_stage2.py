import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from stage2.io_stage1 import load_stage1_result
from stage2.optim_theta import optimize_theta
from stage1.roi import roi_meta as get_roi_meta_dict 
from stage1.config_stage1 import CFG_STAGE1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_obj", required=True)
    ap.add_argument("--stage1_npz", required=True)
    ap.add_argument("--depth_obs_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_noise_std", type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rot, trans, K, roi_xywh = load_stage1_result(args.stage1_npz, device=device)
    full_meta = get_roi_meta_dict(CFG_STAGE1["img_wh"], roi_xywh)
    depth_obs = np.load(args.depth_obs_npy)
    depth_obs_t = torch.from_numpy(depth_obs).float().to(device)

    # 读取 GT
    GT_NPZ = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    gt_data = np.load(GT_NPZ, allow_pickle=True)
    
    gt_pose = gt_data['poses'] # 应该是 (1, 46)
    gt_beta = gt_data['betas'] # 应该是 (1, 10)
    
    # 强制检查维度
    if gt_pose.shape[1] != 46:
        print(f"[Warning] GT Pose shape is {gt_pose.shape}, expected (1, 46). Truncating or Padding.")
        # 如果大了截断，小了补零
        new_pose = np.zeros((1, 46), dtype=np.float32)
        valid_len = min(gt_pose.shape[1], 46)
        new_pose[:, :valid_len] = gt_pose[:, :valid_len]
        gt_pose = new_pose

    optimize_theta(
        depth_obs=depth_obs_t,
        K=K,
        roi_meta_dict=full_meta,
        roi_xywh=roi_xywh,
        rot_tensor=rot,
        trans_tensor=trans,
        gt_pose_np=gt_pose,
        gt_beta_np=gt_beta,
        out_dir=args.out_dir,
        device=device,
        init_noise_std=args.init_noise_std
    )

if __name__ == "__main__":
    main()
