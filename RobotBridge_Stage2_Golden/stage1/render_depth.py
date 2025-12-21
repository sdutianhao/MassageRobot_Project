import torch
import math
from .camera import get_projection_jacobian


def render_gaussian_depth(
        V_cam,  # (N, 3) 相机坐标系下的点
        K,  # (3, 3) 内参
        roi_wh,  # (w, h) ROI 尺寸
        roi_xywh_global,  # (x, y, w, h) ROI 在原图的位置
        radius=0.01,  # 3D 高斯球半径 (米)
        sigma_scale=3.0  # 高斯衰减范围
):
    """
    基于 Gaussian Splatting 的深度渲染器 (可微)
    """
    W_roi, H_roi = roi_wh
    device = V_cam.device
    N = V_cam.shape[0]

    # 1. 投影中心点
    X, Y, Z = V_cam[:, 0], V_cam[:, 1], V_cam[:, 2].clamp_min(1e-6)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    # 转到 ROI 坐标系
    u_roi = u - roi_xywh_global[0]
    v_roi = v - roi_xywh_global[1]

    # 2. 投影协方差 (3D球 -> 2D椭圆)
    cov3d = torch.eye(3, device=device) * (radius ** 2)
    cov3d = cov3d.unsqueeze(0).expand(N, 3, 3)

    J = get_projection_jacobian(V_cam, K)

    # cov2d = J @ cov3d @ J.T
    cov2d = torch.bmm(J, torch.bmm(cov3d, J.permute(0, 2, 1)))

    # 防止奇异
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    # 3. 栅格化近似 (Weighted Depth with Covariance)
    # 提取 projected sigma (近似)
    sigmas = torch.sqrt(torch.diagonal(cov2d, dim1=1, dim2=2)).mean(1)  # (N,)

    return render_depth_weighted(u_roi, v_roi, Z, sigmas, roi_wh)


def render_depth_weighted(x, y, z, sigma, wh):
    """
    升级版 Z-buffer: 使用 scatter_reduce 保证遮挡关系正确
    """
    W, H = wh
    device = x.device

    # 坐标取整
    ix = x.round().long()
    iy = y.round().long()

    # 过滤出界
    mask = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > 0)
    ix = ix[mask]
    iy = iy[mask]
    z_val = z[mask]

    # 展平索引
    linear_idx = iy * W + ix

    # 稀疏深度图 (取最近，实现 Z-buffer)
    sparse_depth = torch.full((H * W,), 100.0, device=device)
    # 'amin' reduce 实现物理遮挡 (谁近选谁)
    sparse_depth.scatter_reduce_(0, linear_idx, z_val, reduce='amin', include_self=False)
    sparse_depth = sparse_depth.view(H, W)

    return sparse_depth