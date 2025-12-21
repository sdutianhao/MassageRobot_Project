import torch


def build_K(fx, fy, cx, cy, device=None, dtype=torch.float32):
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype
    )
    return K


def axis_angle_to_matrix(w):
    """
    w: (..., 3) axis-angle
    return: (..., 3, 3) rotation matrix
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / theta

    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)

    K = torch.stack([
        torch.stack([ zero, -kz,   ky], dim=-1),
        torch.stack([  kz,  zero, -kx], dim=-1),
        torch.stack([ -ky,  kx,  zero], dim=-1),
    ], dim=-2)

    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]

    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def apply_rigid(V, rotvec, trans):
    """
    V: (N,3) world coords
    rotvec: (3,)
    trans: (3,)
    """
    R = axis_angle_to_matrix(rotvec[None])[0]
    return V @ R.T + trans[None]


def project_points(V_cam, K):
    """
    V_cam: (N,3) camera coords
    """
    X, Y, Z = V_cam[:, 0], V_cam[:, 1], V_cam[:, 2].clamp_min(1e-6)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return torch.stack([u, v], dim=-1), Z


def get_projection_jacobian(V_cam, K):
    """
    [新增] 计算投影的仿射近似雅可比矩阵 J
    用于将 3D 协方差传播到 2D
    J = [fx/Z, 0, -(fx*X)/Z^2]
        [0, fy/Z, -(fy*Y)/Z^2]
    """
    X, Y, Z = V_cam[:, 0], V_cam[:, 1], V_cam[:, 2].clamp_min(1e-6)
    fx, fy = K[0, 0], K[1, 1]

    # 构造 J 矩阵 (N, 2, 3)
    zeros = torch.zeros_like(Z)

    row0 = torch.stack([fx / Z, zeros, -(fx * X) / (Z * Z)], dim=-1)
    row1 = torch.stack([zeros, fy / Z, -(fy * Y) / (Z * Z)], dim=-1)

    J = torch.stack([row0, row1], dim=1)
    return J
