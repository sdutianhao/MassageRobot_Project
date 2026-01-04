import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1.camera import project_points


def roi_vertex_mask(V_cam: torch.Tensor, K: torch.Tensor, roi_xywh: list):
    """
    V_cam: (N,3) camera/world-consistent coords used by project_points
    roi_xywh: [rx, ry, rw, rh]
    return mask (N,) bool where projected pixels fall inside ROI and z>0
    """
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    pix, _ = project_points(V_cam, K)  # (N,2) in full image coords
    x = pix[:, 0]
    y = pix[:, 1]
    z = V_cam[:, 2]
    m = (z > 1e-6) & (x >= rx) & (x < rx + rw) & (y >= ry) & (y < ry + rh)
    return m


def save_roi_overlay_png(
    V_pred_cam: torch.Tensor,
    V_gt_cam: torch.Tensor,
    faces_np: np.ndarray,
    centroids_pred_cam: torch.Tensor,
    centroids_gt_cam: torch.Tensor,
    K: torch.Tensor,
    roi_xywh: list,
    out_png: str,
    ellipsoid_rot_mats: torch.Tensor = None,
    title: str = ""
):
    """
    画 ROI 内局部 3D：pred/gt 网格叠加 + 椭球中心点（用三角面中心点表示）。
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    m_pred = roi_vertex_mask(V_pred_cam, K, roi_xywh)
    m_gt = roi_vertex_mask(V_gt_cam, K, roi_xywh)

    Vp = V_pred_cam[m_pred].detach().cpu().numpy()
    Vg = V_gt_cam[m_gt].detach().cpu().numpy()

    Cp = centroids_pred_cam.detach().cpu().numpy()
    Cg = centroids_gt_cam.detach().cpu().numpy()

    # 仅画 ROI 内的中心点（用其投影是否落入 ROI 判断）
    mCp = roi_vertex_mask(torch.from_numpy(Cp).to(V_pred_cam.device), K, roi_xywh).detach().cpu().numpy()
    mCg = roi_vertex_mask(torch.from_numpy(Cg).to(V_gt_cam.device), K, roi_xywh).detach().cpu().numpy()
    Cp = Cp[mCp]
    Cg = Cg[mCg]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    if Vp.shape[0] > 0:
        ax.scatter(Vp[:, 0], Vp[:, 1], Vp[:, 2], s=1, label="pred(roi)")
    if Vg.shape[0] > 0:
        ax.scatter(Vg[:, 0], Vg[:, 1], Vg[:, 2], s=1, label="gt(roi)")

    # “椭球”：用中心点散点表示（你现在 Stage2 固定椭球参数，只需要看中心分布）
    if Cp.shape[0] > 0:
        ax.scatter(Cp[:, 0], Cp[:, 1], Cp[:, 2], s=6, marker="o", label="ellipsoid centers (pred)")
    if Cg.shape[0] > 0:
        ax.scatter(Cg[:, 0], Cg[:, 1], Cg[:, 2], s=6, marker="x", label="ellipsoid centers (gt)")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def save_overlay_ellipsoids_gt_png(
    v_gt_cam: torch.Tensor,
    centers_cam: torch.Tensor,
    ellipsoid_log_scales: torch.Tensor,
    K: torch.Tensor,
    roi_xywh: list,
    roi_wh,
    out_png: str,
    ellipsoid_rot_mats: torch.Tensor = None,
    title: str = "",
    max_gt_points: int = 20000,
    max_ellipsoids: int = None,
    ellipsoid_level: float = 1.0,
):
    """
    2D overlay in ROI pixel coordinates:
      - GT mesh vertices projected as scatter (true human)
      - Ellipsoids projected as 2D ellipses using (sx, sy) and depth z

    All inputs are stage-agnostic:
      v_gt_cam: (N,3) in camera frame
      centers_cam: (M,3) in camera frame (typically triangle centroids)
      ellipsoid_log_scales: (M,3) log scales (sx,sy,sz)
      K: (3,3)
      roi_xywh: [rx, ry, rw, rh] in full image coords
      roi_wh: (w,h) of ROI depth image
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    roi_w, roi_h = int(roi_wh[0]), int(roi_wh[1])

    # robust: if matplotlib backends missing, do not crash pipeline
    try:
        from matplotlib.patches import Ellipse
    except Exception:
        return

    device = v_gt_cam.device
    K = K.to(device)

    def proj_roi(X):
        pix, _ = project_points(X, K)  # full image coords
        u = pix[:, 0] - rx
        v = pix[:, 1] - ry
        z = X[:, 2]
        return u, v, z

    # ---- GT vertices (subsample for speed) ----
    V = v_gt_cam
    if V.dim() == 3 and V.shape[0] == 1:
        V = V[0]
    if V.shape[0] > int(max_gt_points):
        idx = torch.randperm(V.shape[0], device=V.device)[: int(max_gt_points)]
        Vp = V[idx]
    else:
        Vp = V

    u_gt, v_gt, z_gt = proj_roi(Vp)
    m_gt = (z_gt > 1e-6) & (u_gt >= 0) & (u_gt < roi_w) & (v_gt >= 0) & (v_gt < roi_h)
    u_gt = u_gt[m_gt].detach().cpu().numpy()
    v_gt = v_gt[m_gt].detach().cpu().numpy()

    # ---- Ellipsoids ----
    C = centers_cam
    u_c, v_c, z_c = proj_roi(C)
    m_c = (z_c > 1e-6) & (u_c >= 0) & (u_c < roi_w) & (v_c >= 0) & (v_c < roi_h)

    u_c = u_c[m_c]
    v_c = v_c[m_c]
    z_c = z_c[m_c]
    ls = ellipsoid_log_scales[m_c]

    if max_ellipsoids is not None and int(u_c.numel()) > int(max_ellipsoids):
        sel = torch.randperm(int(u_c.numel()), device=device)[: int(max_ellipsoids)]
        u_c = u_c[sel]
        v_c = v_c[sel]
        z_c = z_c[sel]
        ls = ls[sel]

    s = torch.exp(ls).clamp(min=1e-8)

    fx = float(K[0, 0].detach().cpu().item())
    fy = float(K[1, 1].detach().cpu().item())

    # ellipse width/height (pixels)
    # ellipsoid_level: 1.0 means ~1-sigma, purely for visualization
    w_pix = (2.0 * float(ellipsoid_level) * fx * (s[:, 0] / z_c)).detach().cpu().numpy()
    h_pix = (2.0 * float(ellipsoid_level) * fy * (s[:, 1] / z_c)).detach().cpu().numpy()

    u_c_np = u_c.detach().cpu().numpy()
    v_c_np = v_c.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111)

    ax.set_xlim(0, roi_w)
    ax.set_ylim(roi_h, 0)  # image coords (y down)
    ax.set_aspect("equal")
    ax.axis("off")

    if u_gt.size > 0:
        ax.scatter(u_gt, v_gt, s=0.2, alpha=0.25, linewidths=0)

    for x, y, ww, hh in zip(u_c_np, v_c_np, w_pix, h_pix):
        if ww <= 0 or hh <= 0:
            continue
        if ww > roi_w * 2 or hh > roi_h * 2:
            continue
        e = Ellipse((float(x), float(y)), width=float(ww), height=float(hh),
                    angle=0.0, fill=False, linewidth=0.4, alpha=0.35)
        ax.add_patch(e)

    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
