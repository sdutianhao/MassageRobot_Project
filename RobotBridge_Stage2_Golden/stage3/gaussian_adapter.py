import torch
import torch.nn as nn

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为旋转矩阵 (纯 PyTorch 实现，移除 pytorch3d 依赖)
    Args:
        quaternions: (..., 4) tensor, 格式为 [w, x, y, z] (实部在前)
    Returns:
        (..., 3, 3) 旋转矩阵
    """
    # 归一化四元数以确保数值稳定
    quaternions = quaternions / (quaternions.norm(dim=-1, keepdim=True) + 1e-8)
    
    r, i, j, k = torch.unbind(quaternions, -1)
    
    # 计算旋转矩阵元素
    two_s = 2.0  # 因为已经归一化
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

class GaussianMicroSkin(nn.Module):
    def __init__(self, base_verts, base_faces, init_scale=0.005):
        """
        Args:
            base_verts: (N, 3) Stage 2 优化后的固定 SKEL 顶点
            base_faces: (F, 3) 面片索引，用于计算邻接关系 (Laplacian)
            init_scale: 初始椭球半径 (米)
        """
        super().__init__()
        self.num_gaussians = base_verts.shape[0]
        
        # 注册不可训练的基底 (Stage 2 Result)
        self.register_buffer("base_mu", base_verts.detach().clone())
        self.register_buffer("faces", base_faces.detach().clone())
        
        # --- 优化参数 ---
        # 1. 位移场 d (Displacement), 初始化为 0
        self.displacement = nn.Parameter(torch.zeros_like(base_verts))
        
        # 2. 旋转 q (Rotation quaternion), 初始化为单位四元数 [1, 0, 0, 0]
        # 用于旋转椭球方向，但不改变其大小 (行列式不变)
        self.rotation_q = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.num_gaussians, 1)
        )
        
        # --- 固定参数 ---
        # 3. 缩放 S (Scale), 固定以保持行列式不变
        self.register_buffer(
            "fixed_scale", 
            torch.ones(self.num_gaussians, 3) * init_scale
        )
        
        # 4. 透明度 alpha, 固定为 1.0
        self.fixed_alpha = 1.0

        # 预计算邻接列表用于平滑正则
        self._build_adjacency()

    def _build_adjacency(self):
        # 简单的邻接表构建，用于 Laplacian Loss
        # 实际工程中可缓存为稀疏矩阵
        edges = torch.cat([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], dim=0)
        self.edges = edges  # (E, 2)

    def get_gaussian_params(self):
        """
        计算当前的世界坐标高斯参数
        Returns:
            mu: (N, 3) 中心
            sigma: (N, 3, 3) 协方差
        """
        # mu = v_rest + d
        current_mu = self.base_mu + self.displacement
        
        # Sigma = R * S^2 * R^T
        # 使用本地实现的 quaternion_to_matrix
        R = quaternion_to_matrix(self.rotation_q)
        S_sq = torch.diag_embed(self.fixed_scale ** 2)
        current_sigma = torch.bmm(R, torch.bmm(S_sq, R.transpose(1, 2)))
        
        return current_mu, current_sigma

    def compute_laplacian_loss(self):
        """
        计算位移场的拉普拉斯平滑损失 L_smooth
        L = sum || d_i - d_j ||^2
        """
        d = self.displacement
        v1 = d[self.edges[:, 0]]
        v2 = d[self.edges[:, 1]]
        loss = torch.mean(torch.norm(v1 - v2, dim=1) ** 2)
        return loss
