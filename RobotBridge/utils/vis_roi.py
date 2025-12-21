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
