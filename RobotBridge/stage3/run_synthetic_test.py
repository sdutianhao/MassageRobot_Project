import torch
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage2.skel_adapter import SkelAdapter
from stage3.gaussian_adapter import GaussianSkinModel
from stage1.render_depth import render_gaussian_depth
# 复用 Stage 2 的加载逻辑，保证数据同源
try:
    from stage2.io_stage1 import load_stage1_result
except ImportError:
    load_stage1_result = None

def get_next_run_dir(base_root="output"):
    os.makedirs(base_root, exist_ok=True)
    today_str = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join(base_root, f"stage3_{today_str}_run*")
    existing_dirs = glob.glob(pattern)
    run_id = 1
    if existing_dirs:
        ids = [int(d.split("_run")[-1]) for d in existing_dirs if "_run" in d]
        if ids: run_id = max(ids) + 1
    full_path = os.path.join(base_root, f"stage3_{today_str}_run{run_id}")
    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_camera_params(args, device):
    """
    优先读取真实 Stage 1 参数。
    只有在没有任何输入时，才回退到 Standard Synthetic View (Default)。
    """
    # 1. 尝试从命令行参数读取
    if args.stage1_npz and os.path.exists(args.stage1_npz) and load_stage1_result is not None:
        print(f"[Setup] Loading Stage 1 Params from: {args.stage1_npz}")
        rot, trans, K, roi_xywh = load_stage1_result(args.stage1_npz, device=device)
        return rot, trans, K, roi_xywh

    # 2. 尝试从默认路径读取
    default_npz = "experiments/synthetic_case_001/stage1_results.npz"
    if os.path.exists(default_npz) and load_stage1_result is not None:
        print(f"[Setup] Loading Stage 1 Params from default: {default_npz}")
        rot, trans, K, roi_xywh = load_stage1_result(default_npz, device=device)
        return rot, trans, K, roi_xywh

    # 3. 如果都没有，假设是标准的 Synthetic Case 001 视角 (Hardcoded Defaults)
    # 警告：必须确保这里的参数与生成 depth.npy 的参数一致！
    print(f"[Warning] No Stage 1 NPZ found. Using STANDARD SYNTHETIC DEFAULTS.")
    rot = torch.eye(3).to(device)
    trans = torch.tensor([[0.0, 0.0, 0.8]]).to(device) # z=0.8m
    K = torch.tensor([[800.0, 0.0, 256.0],
                      [0.0, 800.0, 256.0],
                      [0.0, 0.0, 1.0]]).to(device)
    roi_xywh = np.array([128, 128, 256, 256], dtype=np.int32) # Center Crop
    return rot, trans, K, roi_xywh

def save_comparison_ply(v_gt, v_pred, faces, filename):
    vb = v_gt.detach().cpu().numpy()
    vg = v_pred.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()
    nv = vb.shape[0]
    nf = f.shape[0]
    
    with open(filename, 'w') as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {nv * 2}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {nf * 2}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        for i in range(nv): fh.write(f"{vb[i,0]:.4f} {vb[i,1]:.4f} {vb[i,2]:.4f} 0 0 255\n")
        for i in range(nv): fh.write(f"{vg[i,0]:.4f} {vg[i,1]:.4f} {vg[i,2]:.4f} 0 255 0\n")
        for i in range(nf): fh.write(f"3 {f[i,0]} {f[i,1]} {f[i,2]}\n")
        for i in range(nf): fh.write(f"3 {f[i,0]+nv} {f[i,1]+nv} {f[i,2]+nv}\n")
    print(f"[Viz] Saved: {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--stage1_npz", type=str, default=None, help="Path to stage1_results.npz matching the depth map")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = get_next_run_dir("output")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1. 获取相机参数 (Load or Default)
    rot, trans, K, roi_xywh = get_camera_params(args, device)
    
    # 提取 ROI 信息
    rx, ry, rw, rh = roi_xywh
    roi_wh = (int(rw), int(rh))
    
    # --- 关键修复 1: 修正 K 矩阵以适配 ROI 渲染 ---
    # 渲染器只负责画局部图，所以主点 (cx, cy) 必须减去 ROI 的左上角偏移
    K_roi = K.clone()
    K_roi[0, 2] -= rx
    K_roi[1, 2] -= ry
    
    # 2. 加载 GT 骨架
    gt_npz = "experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    gt_data = np.load(gt_npz, allow_pickle=True)
    skel_gen = SkelAdapter(gt_pose_np=gt_data['poses'], gt_beta_np=gt_data['betas'], init_noise_std=0.0, device=device).to(device)
    v_gt = skel_gen.forward_vertices().detach()
    faces = torch.tensor(skel_gen.faces.astype(np.int32)).long().to(device)

    # 3. 构造带噪初始值 (Green Start)
    # 计算哪些点在 ROI 里，只给它们加噪声
    v_world_gt = torch.matmul(v_gt, rot.T) + trans
    v_img = torch.matmul(v_world_gt, K.T) # 用全图 K 投影
    u, v = v_img[:, 0]/v_img[:, 2], v_img[:, 1]/v_img[:, 2]
    roi_mask = (u >= rx) & (u <= rx+rw) & (v >= ry) & (v <= ry+rh)
    
    print(f"[Setup] ROI Vertices: {roi_mask.sum().item()} / {len(v_gt)}")
    
    model = GaussianSkinModel(v_gt, faces, device=device).to(device)
    noise = torch.randn_like(v_gt) * 0.015 # 1.5cm 噪声
    with torch.no_grad():
        model.displacements[roi_mask] = noise[roi_mask]
        
    save_comparison_ply(v_gt, model(), faces, os.path.join(ckpt_dir, "comparison_START.ply"))

    # 4. 准备真值深度
    # 优先用磁盘上的真实深度
    depth_path = "experiments/synthetic_case_001/depth.npy"
    if os.path.exists(depth_path):
        print(f"[Data] Using REAL depth from {depth_path}")
        depth_gt = torch.from_numpy(np.load(depth_path)).float().to(device)
    else:
        print("[Data] Generating SYNTHETIC depth (Self-Check)")
        # 注意：生成 GT 深度也要用修正后的 K_roi
        depth_gt = render_gaussian_depth(v_world_gt, K_roi, roi_wh, radius=0.015)

    # 5. 优化循环
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("\n>>> Start Optimization <<<")
    
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        v_local = model()
        v_world = torch.matmul(v_local, rot.T) + trans
        
        # --- 关键修复 2: 接口调用修正 ---
        # 移除 roi_xywh，使用 K_roi
        depth_pred = render_gaussian_depth(
            v_world, 
            K_roi,     # 修正后的内参
            roi_wh,    # 仅宽高
            radius=0.015,
            sigma_scale=3.0
        )
        
        mask = (depth_gt > 0.01) & (depth_pred > 0.01)
        if mask.sum() > 0:
            loss = torch.nn.functional.mse_loss(depth_pred[mask], depth_gt[mask])
            loss.backward()
            optimizer.step()
        else:
            loss = torch.tensor(0.0)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

    save_comparison_ply(v_gt, model(), faces, os.path.join(ckpt_dir, "comparison_END.ply"))
    print(f"\n[Done] Results -> {ckpt_dir}")

if __name__ == "__main__":
    main()
