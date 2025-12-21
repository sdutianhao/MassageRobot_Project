import torch
import torch.optim as optim
import numpy as np
import argparse
import sys
import os

# 确保能导入同级模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage3.gaussian_adapter import GaussianMicroSkin
from stage3.renderer import compute_depth_map

def run_optimization(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Stage 3] Device: {device}")

    # 1. 数据加载
    stage2_verts_path = os.path.join(args.input_dir, "stage2_verts.npy")
    stage2_faces_path = os.path.join(args.input_dir, "stage2_faces.npy")
    
    if os.path.exists(stage2_verts_path):
        print(f"[Stage 3] Loading Stage 2 output from {stage2_verts_path}")
        verts = torch.from_numpy(np.load(stage2_verts_path)).float().to(device)
        faces = torch.from_numpy(np.load(stage2_faces_path)).long().to(device)
    else:
        print("[Stage 3] Warning: Stage 2 output not found. Generating dummy mesh.")
        verts = torch.randn(6890, 3).to(device) * 0.1
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long).repeat(100, 1).to(device)

    # 加载 GT 深度图
    if os.path.exists(args.gt_depth_path):
        gt_depth = torch.from_numpy(np.load(args.gt_depth_path)).float().to(device)
    else:
        print(f"[Stage 3] Warning: GT Depth not found. Using zeros.")
        gt_depth = torch.zeros((512, 512), device=device)

    H, W = gt_depth.shape

    # 2. 初始化模型
    model = GaussianMicroSkin(verts, faces, init_scale=0.005).to(device)

    # 3. 优化器配置
    optimizer = optim.Adam([
        {'params': model.displacement, 'lr': args.lr_disp},
        {'params': model.rotation_q, 'lr': args.lr_rot}
    ])

    print("[Stage 3] Optimization Start: Tuning Micro-skin...")

    # --- 关键修复：设置合理的相机参数 ---
    K = torch.eye(3).to(device)
    # 假设 FOV 约 60 度，f ~ 500-800
    K[0,0], K[1,1], K[0,2], K[1,2] = 800.0, 800.0, W/2.0, H/2.0
    
    T_cw = torch.eye(4).to(device)
    # ！！！让相机后退 0.5 米，以便看到位于 z=0 的 Mesh ！！！
    # 注意：这只是默认值。在真实运行中，T_cw 应该从文件加载。
    T_cw[2, 3] = 0.5 

    for epoch in range(args.epochs):
        optimizer.zero_grad()

        # Forward
        mu, sigma = model.get_gaussian_params()

        # Rendering
        mu_2d, z_pred, (v_idx, u_idx) = compute_depth_map(mu, sigma, K, T_cw, (H, W))

        # --- Loss 计算 ---
        z_gt_sampled = gt_depth[v_idx, u_idx]
        
        # Mask: 仅在 GT 有效且预测有效的地方计算
        valid_mask = (z_gt_sampled > 0.01) & (z_pred > 0.01)
        
        if valid_mask.sum() > 0:
            l_depth = torch.nn.functional.smooth_l1_loss(z_pred[valid_mask], z_gt_sampled[valid_mask])
            
            l_disp = torch.mean(torch.norm(model.displacement, dim=1)**2)
            l_smooth = model.compute_laplacian_loss()

            loss = l_depth + args.lam_disp * l_disp + args.lam_smooth * l_smooth

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Iter {epoch}: Total={loss.item():.6f} | Depth={l_depth.item():.6f} | Disp={l_disp.item():.6f}")
        else:
            # 如果依然没有重叠，打印警告（仅前几次）
            if epoch < 5:
                print(f"Iter {epoch}: Warning - No valid overlap between Mesh projection and GT Depth.")

    # 4. 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "stage3_micro_skin.pt")
    torch.save({
        'displacement': model.displacement.detach().cpu(),
        'rotation': model.rotation_q.detach().cpu(),
        'sigma': model.get_gaussian_params()[1].detach().cpu()
    }, output_path)
    print(f"[Stage 3] Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='output/stage2')
    parser.add_argument('--gt_depth_path', type=str, default='experiments/synthetic_case_001/depth.npy')
    parser.add_argument('--output_dir', type=str, default='output/stage3')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_disp', type=float, default=5e-3)
    parser.add_argument('--lr_rot', type=float, default=1e-3)
    parser.add_argument('--lam_disp', type=float, default=10.0)
    parser.add_argument('--lam_smooth', type=float, default=50.0)
    
    args = parser.parse_args()
    run_optimization(args)
