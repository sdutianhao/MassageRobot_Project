import os
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from stage2.config_stage2 import CFG_STAGE2
from stage2.skel_adapter import SkelAdapter 
from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import axis_angle_to_matrix
from stage1.render_depth import render_gaussian_depth 
from utils.vis import save_depth_vis

def save_comparison_ply(V_pred, V_gt, Faces, filename):
    if torch.is_tensor(V_pred): V_pred = V_pred.detach().cpu().numpy()
    if torch.is_tensor(V_gt): V_gt = V_gt.detach().cpu().numpy()
    
    mesh_pred = trimesh.Trimesh(vertices=V_pred, faces=Faces)
    mesh_pred.visual.vertex_colors = [255, 0, 0, 150] 
    mesh_gt = trimesh.Trimesh(vertices=V_gt, faces=Faces)
    mesh_gt.visual.vertex_colors = [0, 255, 0, 150] 
    
    scene = trimesh.Scene([mesh_pred, mesh_gt])
    concat_mesh = trimesh.util.concatenate(scene.dump())
    concat_mesh.export(filename)
    print(f"[Vis3D] Saved comparison -> {filename}")

def optimize_theta(
    depth_obs: torch.Tensor,
    K: torch.Tensor,
    roi_meta_dict: dict,
    roi_xywh: list,
    rot_tensor: torch.Tensor,
    trans_tensor: torch.Tensor,
    gt_pose_np: np.ndarray,
    gt_beta_np: np.ndarray,
    out_dir: str,
    device: torch.device,
    init_noise_std: float = 0.0,
):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis_process")
    os.makedirs(vis_dir, exist_ok=True)
    ply_dir = os.path.join(out_dir, "vis_3d")
    os.makedirs(ply_dir, exist_ok=True)

    torch.manual_seed(int(CFG_STAGE2["seed"]))
    
    skel = SkelAdapter('female', gt_pose_np, gt_beta_np, init_noise_std, device).to(device)
    
    if rot_tensor.shape == (3,): R = axis_angle_to_matrix(rot_tensor[None])[0] 
    else: R = rot_tensor
    t = trans_tensor.reshape(1, 3)

    with torch.no_grad():
        skel_gt = SkelAdapter('female', gt_pose_np, gt_beta_np, 0.0, device).to(device)
        V_gt_world = skel_gt.forward_vertices() @ R.T + t
        V_gt_np = V_gt_world.cpu().numpy()

    # 4. 优化器与调度器
    optimizer = torch.optim.Adam([
        {'params': [skel.pose], 'lr': 0.001},
        {'params': [skel.beta], 'lr': 0.001},
        {'params': [skel.trans], 'lr': 0.0005}
    ])
    
    iters = int(CFG_STAGE2["iters"])
    # 增加学习率调度器：在总步数的 80% 处将 LR 减半，最后稳住
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(iters*0.8)], gamma=0.5)

    print_every = int(CFG_STAGE2["print_every"])
    roi_wh = roi_meta_dict["roi_wh"] 

    print(f"[Stage2] Start Heavy-Regulated Optim. iters={iters}, Noise={init_noise_std}")

    for it in range(iters):
        if it == 0:
            with torch.no_grad():
                V_curr = (skel.forward_vertices() @ R.T + t).cpu().numpy()
                save_comparison_ply(V_curr, V_gt_np, skel.faces, os.path.join(ply_dir, "opt_start.ply"))

        optimizer.zero_grad()
        V_skel = skel.forward_vertices()
        V_world = V_skel @ R.T + t
        depth_pred = render_gaussian_depth(V_world, K, roi_wh, roi_xywh, radius=0.025, sigma_scale=3.0)

        # 诊断
        valid_obs = (depth_obs > 1e-4)
        valid_pred = (depth_pred > 1e-4) & (depth_pred < 50.0)
        mask = valid_obs & valid_pred
        
        if it == 0:
             print(f"=== [Diagnostic] Iter 0: Obs Mean={depth_obs[valid_obs].mean():.4f}, Pred Mean={depth_pred[valid_pred].mean():.4f}")

        if mask.sum() == 0: 
            loss_depth = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            diff = (depth_pred - depth_obs)[mask]
            loss_depth = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.01)

        # [核心修改] 大幅提高正则权重
        # 1. Pose Regularization: 约束它不要偏离初始值（或者零值）太远
        # 这里我们假设初始值(gt_pose_np)是先验知识，所以我们约束它不要偏离 gt_pose_np 太多
        # 但因为 gt_pose_np 本身就是我们的初始化，所以直接对 skel.pose 与 初始值的差做惩罚
        # 注意：skel.pose 初始化时已经加了噪声，所以这里我们用 L2 norm 约束其不要发散
        
        # 简单粗暴方案：直接约束 Pose 的绝对值大小，防止数值爆炸
        # 权重提高到 1.0 (原 0.01)
        loss_pose_reg = (skel.pose ** 2).mean() * 1.0 
        
        # 2. Beta Regularization
        # 权重提高到 0.1 (原 0.01)
        loss_beta_reg = (skel.beta ** 2).mean() * 0.1

        loss = loss_depth + loss_pose_reg + loss_beta_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(skel.parameters(), max_norm=0.5) # 梯度裁剪更严格
        optimizer.step()
        scheduler.step()

        if it % print_every == 0 or it == iters - 1:
            print(f"[Stage2][{it:03d}/{iters}] L_total={loss.item():.5f} (Depth={loss_depth.item():.5f}, Reg={loss_pose_reg.item()+loss_beta_reg.item():.5f})")

    with torch.no_grad():
        V_final = (skel.forward_vertices() @ R.T + t).cpu().numpy()
        save_comparison_ply(V_final, V_gt_np, skel.faces, os.path.join(ply_dir, "opt_end.ply"))

    theta_out = {'pose': skel.pose.detach().cpu().numpy(), 'beta': skel.beta.detach().cpu().numpy()}
    np.save(os.path.join(out_dir, "theta_opt.npy"), theta_out)
