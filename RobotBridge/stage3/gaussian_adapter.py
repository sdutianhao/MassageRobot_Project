import torch
import torch.nn as nn

from stage1.camera import apply_rigid, project_points


def _roi_mask_points_cam(points_cam: torch.Tensor, K: torch.Tensor, roi_xywh):
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    if points_cam.dim() == 3 and points_cam.shape[0] == 1:
        points_cam = points_cam[0]
    uv, _ = project_points(points_cam, K)
    if uv.dim() == 3 and uv.shape[0] == 1:
        uv = uv[0]
    u = uv[..., 0]
    v = uv[..., 1]
    z = points_cam[..., 2]
    return (z > 1e-6) & (u >= rx) & (u < rx + rw) & (v >= ry) & (v < ry + rh)


class GaussianSkinModel(nn.Module):
    """
    Stage3（按你的定义）：
    - 固定 θ，不直接优化椭球中心位置
    - 优化：d_k（位移基元参数） + 协方差形状（体积固定）
    - 绿色网格：V' = V + ΔV(d_k, Σ)
    - 红色椭球中心：C_k = centroid(V', face_k) —— 只能由 V' 传递得到
    - ray NLL 用 C_k 与 Σ；权重场 w_ik ∝ density(Σ) 作用在 d_k 上
    """
    def __init__(
        self,
        init_vertices: torch.Tensor,     # (N,3) world 或 (1,N,3)
        faces: torch.Tensor,             # (F,3) long
        K: torch.Tensor,                 # (3,3)
        rot: torch.Tensor,               # (3,3)
        trans: torch.Tensor,             # (1,3) or (3,)
        roi_xywh,
        ellipsoid_s0: float = 0.01,      # 固定大小（10mm）
        max_ellipsoids: int = 5000,
        chunk_k: int = 1024,
        device: str = "cuda",
    ):
        super().__init__()
        if init_vertices.dim() == 3 and init_vertices.shape[0] == 1:
            init_vertices = init_vertices[0]

        self.device = device
        self.register_buffer("base_verts", init_vertices.clone().detach().to(device))
        self.register_buffer("faces", faces.clone().detach().to(device))
        self.register_buffer("K", K.clone().detach().to(device))
        self.register_buffer("rot", rot.clone().detach().to(device))
        self.register_buffer("trans", trans.reshape(1, 3).clone().detach().to(device))
        self.roi_xywh = [float(x) for x in roi_xywh]

        self.ellipsoid_s0 = float(ellipsoid_s0)
        self.max_ellipsoids = int(max_ellipsoids)
        self.chunk_k = int(chunk_k)

        # ---- 固定 ROI 顶点集合（用 base 投影选一次） ----
        with torch.no_grad():
            v_cam0 = apply_rigid(self.base_verts, self.rot, self.trans)
            vmask = _roi_mask_points_cam(v_cam0, self.K, self.roi_xywh)
            roi_v_idx = torch.where(vmask)[0].long()
            if roi_v_idx.numel() == 0:
                roi_v_idx = torch.arange(self.base_verts.shape[0], device=device).long()
            self.register_buffer("roi_v_idx", roi_v_idx)

        # ---- 固定 ROI faces 子集（椭球索引集合），但中心由 V' 质心实时计算 ----
        with torch.no_grad():
            tri0 = self.base_verts[self.faces]                      # (F,3,3)
            c0_world = tri0.mean(dim=1)                             # (F,3)
            c0_cam = apply_rigid(c0_world, self.rot, self.trans)    # (F,3)
            fmask = _roi_mask_points_cam(c0_cam, self.K, self.roi_xywh)
            roi_f_idx = torch.where(fmask)[0].long()
            if roi_f_idx.numel() == 0:
                roi_f_idx = torch.arange(c0_world.shape[0], device=device).long()

            if roi_f_idx.numel() > self.max_ellipsoids:
                sel = torch.linspace(0, roi_f_idx.numel() - 1, steps=self.max_ellipsoids, device=device).long()
                roi_f_idx = roi_f_idx[sel]

            self.register_buffer("roi_f_idx", roi_f_idx)
            self.register_buffer("roi_faces", self.faces[roi_f_idx].clone().detach())     # (K,3)
            # 权重场使用的 anchor（固定）：base face 质心
            self.register_buffer("centers0_world", c0_world[roi_f_idx].clone().detach()) # (K,3)

        K_ell = int(self.roi_faces.shape[0])

        # ---- 可学习参数：d_k 仅作位移基元，不再直接改中心 ----
        self.ellipsoid_disp = nn.Parameter(torch.zeros((K_ell, 3), device=device))       # d_k
        self.ellipsoid_shape_raw = nn.Parameter(torch.zeros((K_ell, 3), device=device)) # shape

        self._last_centers_world = None

    def shape_l(self):
        # enforce l1+l2+l3 = 0 （固定体积/尺度）
        l = self.ellipsoid_shape_raw - self.ellipsoid_shape_raw.mean(dim=1, keepdim=True)
        return l

    def ellipsoid_log_scales(self):
        # log_scales = log(s0) + l, 只改变各向异性，不改变总体积
        log_s0 = torch.log(torch.tensor(self.ellipsoid_s0, device=self.base_verts.device) + 1e-12)
        return log_s0 + self.shape_l()

    def ellipsoid_centers_world(self, vertices_world: torch.Tensor = None):
        # C_k = centroid(V', roi_faces)
        if vertices_world is None:
            vertices_world = self.base_verts
        if vertices_world.dim() == 3 and vertices_world.shape[0] == 1:
            vertices_world = vertices_world[0]
        tri = vertices_world[self.roi_faces]      # (K,3,3)
        return tri.mean(dim=1)                    # (K,3)

    def ellipsoid_centers_cam(self, vertices_world: torch.Tensor = None):
        c_world = self.ellipsoid_centers_world(vertices_world)
        c_cam = apply_rigid(c_world, self.rot, self.trans)
        if c_cam.dim() == 3 and c_cam.shape[0] == 1:
            c_cam = c_cam[0]
        return c_cam

    def forward(self):
        """
        先用 anchor centers0_world + Σ 生成权重场 w_ik，再用 d_k 得到 ΔV。
        然后用 V' 计算椭球中心（供外部 NLL/可视化调用）。
        """
        device = self.base_verts.device
        eps = 1e-12

        V = self.base_verts
        idx = self.roi_v_idx
        V_roi = V[idx]  # (M,3)

        # 权重场 anchor（固定），不允许直接学习中心
        C0 = self.centers0_world                # (K,3)
        d = self.ellipsoid_disp                 # (K,3)
        log_s = self.ellipsoid_log_scales()     # (K,3)
        inv_s2 = torch.exp(-2.0 * log_s)        # (K,3)

        M = int(V_roi.shape[0])
        K_ell = int(C0.shape[0])

        sum_w = torch.zeros((M,), device=device)
        sum_d = torch.zeros((M, 3), device=device)

        for s in range(0, K_ell, self.chunk_k):
            e = min(s + self.chunk_k, K_ell)
            Cc = C0[s:e]            # (kc,3)
            dc = d[s:e]             # (kc,3)
            invc = inv_s2[s:e]      # (kc,3)

            diff = V_roi[:, None, :] - Cc[None, :, :]                 # (M,kc,3)
            quad = (diff * diff * invc[None, :, :]).sum(dim=2)        # (M,kc)
            w = torch.exp(-0.5 * quad)                                # (M,kc)

            sum_w = sum_w + w.sum(dim=1)
            sum_d = sum_d + (w @ dc)

        delta_roi = sum_d / (sum_w[:, None] + eps)  # (M,3)

        delta = torch.zeros_like(V)
        delta[idx] = delta_roi
        V_def = V + delta

        # cache centers from deformed mesh（用于外部查看/调试）
        self._last_centers_world = self.ellipsoid_centers_world(V_def).detach()
        return V_def
