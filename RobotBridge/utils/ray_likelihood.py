import torch
from stage1.camera import project_points


def _sample_valid_roi_pixels(depth_obs: torch.Tensor, num_pix: int):
    mask = (depth_obs > 1e-4) & (depth_obs < 50.0)
    idx = torch.nonzero(mask, as_tuple=False)  # (N,2) [j,i]
    if idx.numel() == 0:
        return None, None, None
    if idx.shape[0] > num_pix:
        sel = torch.randint(0, idx.shape[0], (num_pix,), device=depth_obs.device)
        idx = idx[sel]
    j = idx[:, 0].long()
    i = idx[:, 1].long()
    D = depth_obs[j, i]
    return j, i, D


def _depth_gate_centers_cam(centers_cam: torch.Tensor, K: torch.Tensor, roi_xywh, depth_obs: torch.Tensor, tau: float):
    """单视角深度只约束前表面：过滤掉明显在观测深度后方的中心。tau=允许后退量(米)。"""
    if float(tau) <= 0.0:
        return torch.ones((centers_cam.shape[0],), device=centers_cam.device, dtype=torch.bool)

    rx, ry = float(roi_xywh[0]), float(roi_xywh[1])
    roi_h, roi_w = int(depth_obs.shape[0]), int(depth_obs.shape[1])

    uv, _ = project_points(centers_cam, K)
    u = uv[:, 0] - rx
    v = uv[:, 1] - ry
    z = centers_cam[:, 2]

    in_img = (z > 1e-6) & (u >= 0) & (u < roi_w) & (v >= 0) & (v < roi_h)

    i = torch.round(u).long().clamp(0, roi_w - 1)
    j = torch.round(v).long().clamp(0, roi_h - 1)
    D = depth_obs[j, i]
    valid = (D > 1e-4) & (D < 50.0)

    return in_img & valid & (z <= (D + float(tau)))

def ray_overlap_nll(
    centers_cam: torch.Tensor,
    K: torch.Tensor,
    roi_xywh,
    depth_obs: torch.Tensor,
    sigma_space: float,
    sigma_z: float,
    num_pix: int = 2048,
    eps: float = 1e-12,
    max_ellipsoids: int = 5000,
    ellipsoid_log_scales: torch.Tensor = None,
    use_anisotropic: bool = True,
    learn_ellipsoid: bool = False,
    num_t: int = 5,
    vis_depth_gate: float = 0.0,
):
    """
    概率射线重叠 NLL（像素采样版）

    关键实现点：
    - 不再把观测深度当“硬点”，而是在 D_obs 附近做一个很窄的深度带（由 sigma_z 控制，建议毫米级）
    - 对每个 t 计算：混合密度（椭球）在 x=t*d 处的 log-mean-exp（减 logM）-> 避免“多椭球奖励”
    - 对 t 做加权 log-sum（近似积分），权重来自观测深度分布（以 sigma_z 为宽度）
    """
    device = centers_cam.device

    # 1) 只取落在 ROI 视锥内的中心（投影进 ROI + z>0）
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    uv, _ = project_points(centers_cam, K)
    u = uv[:, 0]
    v = uv[:, 1]
    zc = centers_cam[:, 2]
    m = (zc > 1e-6) & (u >= rx) & (u < rx + rw) & (v >= ry) & (v < ry + rh)

    centers = centers_cam[m]
    if centers.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"num_centers": 0, "num_pix": 0}

    logs = None
    if use_anisotropic and (ellipsoid_log_scales is not None):
        logs = ellipsoid_log_scales[m]

    # subsample ellipsoids（防止爆）
    if centers.shape[0] > max_ellipsoids:
        sel = torch.randint(0, centers.shape[0], (max_ellipsoids,), device=device)
        centers = centers[sel]
        if logs is not None:
            logs = logs[sel]

    # 1.5) 可见性门控：单视角深度只约束前表面，过滤掉明显在观测深度后方的中心（避免背面吸附）
    if float(vis_depth_gate) > 0.0:
        mg = _depth_gate_centers_cam(centers, K, roi_xywh, depth_obs, float(vis_depth_gate))
        centers = centers[mg]
        if logs is not None:
            logs = logs[mg]
        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {"num_centers": 0, "num_pix": 0}

    # 2) 采样 ROI 内有效像素
    j, i, D = _sample_valid_roi_pixels(depth_obs, num_pix=num_pix)
    if j is None:
        return torch.tensor(0.0, device=device, requires_grad=True), {"num_centers": int(centers.shape[0]), "num_pix": 0}

    # 3) 像素 -> 单位射线方向 d(u,v)
    u_full = i.float() + rx
    v_full = j.float() + ry
    ones = torch.ones_like(u_full)
    pix_h = torch.stack([u_full, v_full, ones], dim=1)  # (N,3)

    Kinv = torch.inverse(K)
    dirs = (pix_h @ Kinv.T)  # (N,3)
    dirs = dirs / (torch.linalg.norm(dirs, dim=1, keepdim=True) + 1e-12)  # unit

    # 4) 在 D_obs 附近做“窄带”积分近似：t = D + offset
    #    offsets 用 {-2,-1,0,1,2} * sigma_z （毫米级 sigma_z -> 非常窄）
    if num_t == 5:
        offs = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device) * float(sigma_z)
    else:
        # 通用：均匀覆盖 [-2,2] * sigma_z
        offs = torch.linspace(-2.0, 2.0, steps=int(num_t), device=device) * float(sigma_z)

    # 观测权重（只做相对权重，归一化即可，不需要常数）
    w = torch.exp(-0.5 * (offs / (float(sigma_z) + 1e-12)) ** 2)  # (T,)
    w = w / (w.sum() + 1e-12)
    logw = torch.log(w + 1e-12)  # (T,)

    # 5) 对每个 t，算混合密度在 x=t*d 处的 log-mean-exp
    C = centers  # (M,3)
    M = int(C.shape[0])
    logM = torch.log(torch.tensor(float(M), device=device) + 1e-12)

    if (not use_anisotropic) or (logs is None):
        s2 = float(sigma_space) ** 2
        def log_mix_at_x(x_t):
            diff = x_t[:, None, :] - C[None, :, :]           # (N,M,3)
            quad = (diff * diff).sum(dim=2) / (s2 + 1e-12)   # (N,M)
            log_terms = -0.5 * quad                          # (N,M)
            return torch.logsumexp(log_terms, dim=1) - logM   # (N,)
    else:
        if learn_ellipsoid:
            log_s = logs
        else:
            log_s = logs.detach()
        inv_s2 = torch.exp(-2.0 * log_s)  # (M,3)

        def log_mix_at_x(x_t):
            diff = x_t[:, None, :] - C[None, :, :]                 # (N,M,3)
            quad = (diff * diff * inv_s2[None, :, :]).sum(dim=2)   # (N,M)
            log_terms = -0.5 * quad                                # (N,M)
            return torch.logsumexp(log_terms, dim=1) - logM         # (N,)

    # 6) 积分近似：log ∫ ≈ logsumexp_t ( log_mix(t) + logw(t) )
    #    这里用 logsumexp 实现稳定的加权求和（因为 w 已归一化，是“窄带软深度”）
    log_lh_list = []
    for tt in range(int(offs.numel())):
        t_now = D + offs[tt]                # (N,)
        x_now = dirs * t_now[:, None]       # (N,3)
        log_lh_list.append(log_mix_at_x(x_now) + logw[tt])

    log_lh = torch.logsumexp(torch.stack(log_lh_list, dim=0), dim=0)  # (N,)
    loss = -log_lh.mean()

    stats = {"num_centers": int(C.shape[0]), "num_pix": int(D.shape[0]), "num_t": int(offs.numel())}
    return loss, stats
