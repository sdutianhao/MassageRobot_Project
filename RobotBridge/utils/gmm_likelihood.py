import math
import torch


def _sample_valid_depth_pixels(depth_obs: torch.Tensor, num_pix: int):
    # depth_obs: (H, W) ROI depth
    valid = (depth_obs > 1e-4) & (depth_obs < 50.0)
    idx = torch.nonzero(valid, as_tuple=False)  # (N,2) -> (y,x)
    if idx.numel() == 0:
        return idx
    n = int(idx.shape[0])
    k = min(int(num_pix), n)
    perm = torch.randperm(n, device=depth_obs.device)[:k]
    return idx[perm]


def _backproject_roi_pixels_to_cam(
    depth_obs: torch.Tensor, K: torch.Tensor, roi_xywh, pix_yx: torch.Tensor
):
    # pix_yx: (N,2) in ROI coords (y,x)
    rx, ry = float(roi_xywh[0]), float(roi_xywh[1])

    y = pix_yx[:, 0].float()
    x = pix_yx[:, 1].float()
    z = depth_obs[pix_yx[:, 0], pix_yx[:, 1]].float()

    u = x + rx
    v = y + ry
    ones = torch.ones_like(u)

    pix_h = torch.stack([u, v, ones], dim=1)  # (N,3)
    Kinv = torch.inverse(K)
    dirs = pix_h @ Kinv.t()                   # (N,3), dirs[...,2]==1
    X = dirs * z.unsqueeze(1)                 # (N,3)
    return X


def _filter_centers_in_roi(centers_cam: torch.Tensor, K: torch.Tensor, roi_xywh):
    # 与 ray_likelihood 对齐：只取投影进 ROI 且 z>0 的中心
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    z = centers_cam[:, 2]
    eps = 1e-12
    xn = centers_cam[:, 0] / (z + eps)
    yn = centers_cam[:, 1] / (z + eps)

    u = K[0, 0] * xn + K[0, 1] * yn + K[0, 2]
    v = K[1, 0] * xn + K[1, 1] * yn + K[1, 2]

    m = (z > 1e-6) & (u >= rx) & (u < rx + rw) & (v >= ry) & (v < ry + rh)
    return m


def _depth_gate_centers_cam(centers_cam: torch.Tensor, K: torch.Tensor, roi_xywh, depth_obs: torch.Tensor, tau: float, eps: float = 1e-12):
    """单视角深度只约束前表面：过滤掉明显在观测深度后方的中心。tau=允许后退量(米)。"""
    if float(tau) <= 0.0:
        return torch.ones((centers_cam.shape[0],), device=centers_cam.device, dtype=torch.bool)

    rx, ry = float(roi_xywh[0]), float(roi_xywh[1])
    roi_h, roi_w = int(depth_obs.shape[0]), int(depth_obs.shape[1])

    z = centers_cam[:, 2]
    xn = centers_cam[:, 0] / (z + eps)
    yn = centers_cam[:, 1] / (z + eps)

    u_full = K[0, 0] * xn + K[0, 1] * yn + K[0, 2]
    v_full = K[1, 0] * xn + K[1, 1] * yn + K[1, 2]

    u = u_full - rx
    v = v_full - ry

    in_img = (z > 1e-6) & (u >= 0) & (u < roi_w) & (v >= 0) & (v < roi_h)

    i = torch.round(u).long().clamp(0, roi_w - 1)
    j = torch.round(v).long().clamp(0, roi_h - 1)
    D = depth_obs[j, i]
    valid = (D > 1e-4) & (D < 50.0)

    return in_img & valid & (z <= (D + float(tau)))

def gmm_surface_nll(
    centers_cam: torch.Tensor,                 # (M,3)
    K: torch.Tensor,                           # (3,3)
    roi_xywh,
    depth_obs: torch.Tensor,                   # (H,W)
    ellipsoid_rot_mats: torch.Tensor = None,      # (M,3,3) local->cam
    num_pix: int = 2048,
    max_ellipsoids: int = 5000,
    ellipsoid_log_scales: torch.Tensor = None, # (M,3) log(sx,sy,sz)
    use_anisotropic: bool = True,
    sigma_mult: float = 0.05,                  # “搜索半径”(米)，作为附加各向同性方差
    chunk_k: int = 1024,
    vis_depth_gate: float = 0.0,
):
    device = depth_obs.device

    M0 = int(centers_cam.shape[0])
    if M0 == 0:
        stats = {"num_centers": 0, "num_pix": 0, "num_t": 1}
        return torch.zeros([], device=device, requires_grad=True), stats

    # 1) 只取 ROI 视锥内中心（与旧 ray 项一致）
    mask = _filter_centers_in_roi(centers_cam, K, roi_xywh)
    centers = centers_cam[mask]
    if centers.numel() == 0:
        stats = {"num_centers": 0, "num_pix": 0, "num_t": 1}
        return torch.zeros([], device=device, requires_grad=True), stats

    logs = None
    if use_anisotropic and (ellipsoid_log_scales is not None):
        logs = ellipsoid_log_scales[mask]
    rotm = None
    if ellipsoid_rot_mats is not None:
        rotm = ellipsoid_rot_mats[mask]

    # 2) subsample ellipsoids
    M = int(centers.shape[0])
    if M > int(max_ellipsoids):
        sel = torch.randint(0, M, (int(max_ellipsoids),), device=device)
        centers = centers[sel]
        if logs is not None:
            logs = logs[sel]
        if rotm is not None:
            rotm = rotm[sel]
        M = int(centers.shape[0])

    # 2.5) 可见性门控：单视角深度只约束前表面，过滤掉明显在观测深度后方的中心（避免背面吸附）
    if float(vis_depth_gate) > 0.0:
        mg = _depth_gate_centers_cam(centers, K, roi_xywh, depth_obs, float(vis_depth_gate))
        centers = centers[mg]
        if logs is not None:
            logs = logs[mg]
        if rotm is not None:
            rotm = rotm[mg]
        if centers.numel() == 0:
            stats = {"num_centers": 0, "num_pix": 0, "num_t": 1}
            return torch.zeros([], device=device, requires_grad=True), stats
        M = int(centers.shape[0])

    # 3) sample valid depth pixels -> 3D points
    pix_yx = _sample_valid_depth_pixels(depth_obs, int(num_pix))
    if pix_yx.numel() == 0:
        stats = {"num_centers": int(M), "num_pix": 0, "num_t": 1}
        return torch.zeros([], device=device, requires_grad=True), stats

    X = _backproject_roi_pixels_to_cam(depth_obs, K, roi_xywh, pix_yx)  # (N,3)
    N = int(X.shape[0])

    # 4) GMM likelihood (稳定的分块 logsumexp)
    log2pi = math.log(2.0 * math.pi)
    logM = math.log(float(M))
    logsum = torch.full((N,), -1e30, device=device)

    search_var = max(float(sigma_mult), 1e-8) ** 2

    if use_anisotropic and (logs is not None):
        var_model = torch.exp(2.0 * logs)          # (M,3)
        var_eff = var_model + search_var           # (M,3) 关键：方差加法（卷积）
        inv_s2 = 1.0 / (var_eff + 1e-12)           # (M,3)
        log_det = torch.log(var_eff + 1e-12).sum(dim=1)  # (M,)
        const = 0.5 * (3.0 * log2pi + log_det)     # (M,)

        for st in range(0, M, int(chunk_k)):
            ed = min(M, st + int(chunk_k))
            c = centers[st:ed]                     # (m,3)
            inv = inv_s2[st:ed]                    # (m,3)
            ce = const[st:ed]                      # (m,)

            diff = X[:, None, :] - c[None, :, :]                 # (N,m,3)
            if rotm is not None:
                Rc = rotm[st:ed]                                  # (m,3,3) local->cam
                diff_l = torch.einsum('nmi,mij->nmj', diff, Rc)    # (N,m,3) in local
                quad = (diff_l * diff_l * inv[None, :, :]).sum(dim=2)
            else:
                quad = (diff * diff * inv[None, :, :]).sum(dim=2)
            log_phi = -0.5 * quad - ce[None, :]                  # (N,m)

            log_chunk = torch.logsumexp(log_phi, dim=1)          # (N,)
            logsum = torch.logaddexp(logsum, log_chunk)

    else:
        sigma2 = search_var
        inv_sigma2 = 1.0 / (sigma2 + 1e-12)
        const = 0.5 * (3.0 * log2pi + 3.0 * math.log(sigma2 + 1e-12))

        for st in range(0, M, int(chunk_k)):
            ed = min(M, st + int(chunk_k))
            c = centers[st:ed]                                   # (m,3)
            diff = X[:, None, :] - c[None, :, :]                 # (N,m,3)
            quad = (diff * diff).sum(dim=2) * inv_sigma2         # (N,m)
            log_phi = -0.5 * quad - const                        # (N,m)

            log_chunk = torch.logsumexp(log_phi, dim=1)          # (N,)
            logsum = torch.logaddexp(logsum, log_chunk)

    logp = logsum - logM
    nll = -logp.mean()

    stats = {"num_centers": int(M), "num_pix": int(N), "num_t": 1}
    return nll, stats
