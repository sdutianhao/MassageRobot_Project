import torch

def compute_depth_map(mu, sigma, K, T_cw, img_size):
    """
    基于 Gaussian Splatting 的深度渲染 (PyTorch Native 实现)
    Args:
        mu: (N, 3) 世界坐标中心
        sigma: (N, 3, 3) 世界坐标协方差
        K: (3, 3) 内参
        T_cw: (4, 4) 世界转相机外参
        img_size: (H, W)
    Returns:
        mu_2d: (N, 2) 投影坐标
        z: (N,) 深度值
        pixel_indices: (v_int, u_int) 对应的像素索引
    """
    H, W = img_size
    device = mu.device
    
    # 1. World -> Camera 变换
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    
    mu_cam = (torch.matmul(R, mu.T) + t[:, None]).T  # (N, 3)
    z = mu_cam[:, 2] # Depth
    
    # 剔除视锥体外的点 (z < 5cm)
    valid_mask = z > 0.05
    # 注意：为了保持梯度流，这里暂时保留所有点，在 Loss 计算时再做 mask
    # 如果显存紧张，可以在这里应用 mask，但需要处理索引对应关系
    
    # 2. 投影中心 (仅需要中心点进行稀疏深度监督)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x, y = mu_cam[:, 0], mu_cam[:, 1]
    
    u_proj = fx * x / z + cx
    v_proj = fy * y / z + cy
    mu_2d = torch.stack([u_proj, v_proj], dim=1) # (N, 2)
    
    # 3. 像素坐标取整 (用于索引 GT)
    u_int = torch.clamp(u_proj.long(), 0, W-1)
    v_int = torch.clamp(v_proj.long(), 0, H-1)
    
    return mu_2d, z, (v_int, u_int)
