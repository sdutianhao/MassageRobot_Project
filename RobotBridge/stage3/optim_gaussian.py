import torch
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stage3.gaussian_adapter import GaussianMicroSkin
from stage3.renderer import compute_depth_map # 假设您有这个渲染函数

def save_comparison_ply(gt_v, pred_v, faces, filename):
    """
    可视化: GT(蓝) vs Pred(绿)
    完全复刻 Stage 2 的可视化逻辑
    """
    gv = gt_v.detach().cpu().numpy()
    pv = pred_v.detach().cpu().numpy()
    fs = faces.detach().cpu().numpy()
    nv = gv.shape[0] # 应该是 6322
    nf = fs.shape[0]

    with open(filename, 'w') as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {nv * 2}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {nf * 2}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        
        # 1. 写入 GT (Blue)
        for i in range(nv):
            fh.write(f"{gv[i,0]:.6f} {gv[i,1]:.6f} {gv[i,2]:.6f} 0 0 255\n")
        # 2. 写入 Pred (Green)
        for i in range(nv):
            fh.write(f"{pv[i,0]:.6f} {pv[i,1]:.6f} {pv[i,2]:.6f} 0 255 0\n")
        
        # 3. 写入 Faces (GT)
        for i in range(nf):
            fh.write(f"3 {fs[i,0]} {fs[i,1]} {fs[i,2]}\n")
        # 4. 写入 Faces (Pred, index offset +nv)
        for i in range(nf):
            fh.write(f"3 {fs[i,0]+nv} {fs[i,1]+nv} {fs[i,2]+nv}\n")
            
    print(f"[Viz] Saved: {filename}")

def run_optimization(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. 数据准备 (Mock Generator) ---
    from stage3.mock_generator import MockMeshGenerator
    gen = MockMeshGenerator(device)
    
    # 路径硬编码以匹配您的项目结构
    GT_NPZ = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    STAGE1_NPZ = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/stage1_results.npz" # 假设名，如果不同请修改
    if not os.path.exists(STAGE1_NPZ):
         # 尝试从 args 获取，或者 fallback
         STAGE1_NPZ = args.stage1_npz if hasattr(args, 'stage1_npz') else "output/stage1/results.npz"
    
    # 获取数据: v_gt(真值), v_init(ROI带噪)
    v_gt, v_init, faces, (rot, trans, K, roi_xywh) = gen.generate_data(STAGE1_NPZ, GT_NPZ)
    
    # --- 2. 加载真实深度图 (Obs) ---
    depth_obs_np = np.load(args.gt_depth_path)
    depth_obs = torch.from_numpy(depth_obs_np).float().to(device)
    
    # --- 3. 初始化 Stage 3 模型 ---
    # 使用带噪的 v_init 初始化高斯皮肤
    model = GaussianMicroSkin(v_init, faces).to(device)
    optimizer = optim.Adam([{'params': model.displacement, 'lr': args.lr_disp}])
    
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 保存初始状态 (Comparison)
    save_comparison_ply(v_gt, v_init, faces, os.path.join(ckpt_dir, "comparison_START.ply"))
    
    # --- 4. 优化循环 (使用真实深度更新) ---
    print("[Stage 3] Optimizing using REAL DEPTH observation...")
    img_h, img_w = depth_obs.shape
    
    # 构建相机矩阵 4x4
    T_cw = torch.eye(4).to(device)
    T_cw[:3, :3] = rot
    T_cw[:3, 3] = trans
    
    # 模拟高斯协方差 (简化)
    sigma_init = torch.eye(3).unsqueeze(0).repeat(v_gt.shape[0], 1, 1).to(device) * (0.005**2)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        # 获取当前预测顶点
        mu, _ = model.get_gaussian_params()
        
        # 渲染当前深度 (Projected Depth)
        # 注意: 这里的 render 需要支持梯度反传
        _, depth_pred, (v_coords, u_coords) = compute_depth_map(mu, sigma_init, K, T_cw, (img_h, img_w))
        
        # 计算 Loss: 只在有观测深度的地方计算 (Masked MSE)
        # 获取对应像素的真实深度
        target_z = depth_obs[v_coords, u_coords]
        
        # 过滤无效深度 (假设 depth=0 是背景)
        valid_mask = (target_z > 0.1)
        
        if valid_mask.sum() > 0:
            loss = torch.nn.functional.mse_loss(depth_pred[valid_mask], target_z[valid_mask])
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.6f}")

    # --- 5. 保存结果 ---
    final_mu, _ = model.get_gaussian_params()
    save_comparison_ply(v_gt, final_mu, faces, os.path.join(ckpt_dir, "comparison_END.ply"))
    print(f"[Done] Check {ckpt_dir} for comparison PLY files.")

if __name__ == "__main__":
    pass
