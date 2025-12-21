import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_depth_vis(depth_data, out_path, vmin=None, vmax=None):
    """
    将深度图保存为彩色可视化图片 (0值透明或黑色)
    depth_data: tensor or numpy array (H, W)
    """
    if isinstance(depth_data, torch.Tensor):
        d = depth_data.detach().cpu().numpy()
    else:
        d = depth_data.copy()

    # 1. 制作掩膜：过滤掉背景 (0 或 >50 的极大值)
    valid_mask = (d > 1e-4) & (d < 50.0)
    
    if valid_mask.sum() == 0:
        print(f"[Vis] Warning: Depth map is empty! ({out_path})")
        return

    # 2. 归一化颜色范围
    valid_d = d[valid_mask]
    if vmin is None: vmin = valid_d.min()
    if vmax is None: vmax = valid_d.max()

    # 3. 使用 matplotlib 绘图
    plt.figure(figsize=(5, 5))
    #以此背景设为黑色
    plt.imshow(d, cmap='magma', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(label='Depth (m)')
    plt.title(f"Depth Vis\nRange: [{vmin:.3f}, {vmax:.3f}] m")
    
    # 将无效区域设为黑色
    current_cmap = plt.cm.get_cmap('magma')
    current_cmap.set_bad(color='black')
    
    # 保存
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[Vis] Saved depth visualization -> {out_path}")

def save_overlay(img_rgb, img_depth, out_path):
    # 如果未来需要 RGB + Depth 叠加，预留位置
    pass
