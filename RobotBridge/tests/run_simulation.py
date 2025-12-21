import torch
import torch.optim as optim
import sys
import os

# 路径适配
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage3.gaussian_adapter import GaussianMicroSkin
from stage3.renderer import compute_depth_map
from stage3.mock_generator import MockMeshGenerator

def save_obj(verts, faces, filename):
    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Simulation] Device: {device}")
    
    generator = MockMeshGenerator(device)
    output_dir = "output/simulation_test"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 制造 GT
    print("[1] Creating Ground Truth (Muscle Bump)...")
    base_verts, faces = generator.generate_plane(res=64, size=0.4)
    gt_verts = generator.add_bump_detail(base_verts, height=0.04, radius=0.15)
    save_obj(gt_verts.cpu(), faces.cpu(), f"{output_dir}/0_target_gt.obj")

    # 渲染 GT Depth
    H, W = 512, 512
    K = torch.eye(3).to(device)
    K[0,0], K[1,1], K[0,2], K[1,2] = 800.0, 800.0, W/2.0, H/2.0
    T_cw = torch.eye(4).to(device)
    T_cw[2, 3] = 0.5 
    
    gt_sigma = torch.eye(3).unsqueeze(0).repeat(gt_verts.shape[0], 1, 1).to(device) * (0.002**2)
    _, z_gt_vals, (v_gt, u_gt) = compute_depth_map(gt_verts, gt_sigma, K, T_cw, (H, W))
    gt_depth_map = torch.zeros((H, W), device=device)
    gt_depth_map[v_gt, u_gt] = z_gt_vals
    
    # 2. 模拟 Stage 2 输出 (有噪声的平面)
    print("[2] Simulating Stage 2 Output (Noisy & Flat)...")
    stage2_verts = generator.apply_perturbation(base_verts, noise_level=0.002)
    save_obj(stage2_verts.cpu(), faces.cpu(), f"{output_dir}/0_input_stage2_noisy.obj")

    # 3. Stage 3 优化
    print("[3] Running Stage 3 Optimization...")
    model = GaussianMicroSkin(stage2_verts, faces, init_scale=0.006).to(device)
    optimizer = optim.Adam([
        {'params': model.displacement, 'lr': 0.008},
        {'params': model.rotation_q, 'lr': 0.001}
    ])
    
    for epoch in range(151):
        optimizer.zero_grad()
        mu, sigma = model.get_gaussian_params()
        _, z_pred, (v_p, u_p) = compute_depth_map(mu, sigma, K, T_cw, (H, W))
        
        z_target = gt_depth_map[v_p, u_p]
        mask = (z_target > 0.01)
        
        if mask.sum() > 0:
            l_depth = torch.nn.functional.l1_loss(z_pred[mask], z_target[mask])
            l_smooth = model.compute_laplacian_loss() * 40.0
            l_mag = torch.mean(model.displacement ** 2) * 5.0
            loss = l_depth + l_smooth + l_mag
            loss.backward()
            optimizer.step()
            
            if epoch % 30 == 0:
                print(f"Iter {epoch:03d}: Loss={loss.item():.5f} | DepthErr={l_depth.item():.5f}")

    # 4. 结果
    final_mu, _ = model.get_gaussian_params()
    save_obj(final_mu.detach().cpu(), faces.cpu(), f"{output_dir}/1_result_stage3.obj")
    
    err_init = torch.mean(torch.abs(stage2_verts - gt_verts)).item()
    err_final = torch.mean(torch.abs(final_mu - gt_verts)).item()
    print("-" * 30)
    print(f"Initial Error (Stage 2): {err_init*1000:.2f} mm")
    print(f"Final Error   (Stage 3): {err_final*1000:.2f} mm")
    print(f"Improvement: {(err_init - err_final)*1000:.2f} mm reduced")

if __name__ == "__main__":
    run_test()
