import os
import numpy as np
import torch
import trimesh

from stage2.config_stage2 import CFG_STAGE2
from stage2.skel_adapter import SkelAdapter
from stage1.camera import axis_angle_to_matrix, project_points
from stage1.render_depth import render_gaussian_depth
from utils.vis import save_depth_vis
from utils.metrics import pa_mpjpe
from utils.ply_vis import save_stage3_style_comparison_ply, save_roi_overlay_ply, save_roi_mesh_ellipsoids_ply
from utils.ray_likelihood import ray_overlap_nll
from utils.vis_roi import save_overlay_ellipsoids_gt_png


def save_comparison_ply(V_pred, V_gt, Faces, filename):
    mesh_pred = trimesh.Trimesh(vertices=V_pred, faces=Faces, process=False)
    mesh_gt   = trimesh.Trimesh(vertices=V_gt,   faces=Faces, process=False)

    mesh_pred.visual.vertex_colors = [255, 0, 0, 120]   # red translucent
    mesh_gt.visual.vertex_colors   = [0, 255, 0, 120]   # green translucent

    combined = trimesh.util.concatenate([mesh_gt, mesh_pred])
    combined.export(filename)


def _roi_idx_from_gt(V_gt_cam: torch.Tensor, K: torch.Tensor, roi_xywh):
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    uv, _ = project_points(V_gt_cam, K)
    u = uv[:, 0]
    v = uv[:, 1]
    z = V_gt_cam[:, 2]
    m = (z > 1e-6) & (u >= rx) & (u < rx + rw) & (v >= ry) & (v < ry + rh)
    idx = torch.where(m)[0]
    return idx


def _init_fixed_ellipsoid_log_scales(V_cam: torch.Tensor, faces_t: torch.Tensor, min_s: float = 0.01):
    tri = V_cam[faces_t]  # (F,3,3)
    e01 = torch.linalg.norm(tri[:, 1, :] - tri[:, 0, :], dim=1)
    e12 = torch.linalg.norm(tri[:, 2, :] - tri[:, 1, :], dim=1)
    e20 = torch.linalg.norm(tri[:, 0, :] - tri[:, 2, :], dim=1)
    s_xy = (e01 + e12 + e20) / 3.0
    s_xy = torch.clamp(s_xy * 0.5, min=min_s)
    s_z = torch.clamp(s_xy * 1.0, min=min_s)
    scales = torch.stack([s_xy, s_xy, s_z], dim=1)  # (F,3)
    log_scales = torch.log(scales)
    return log_scales


def optimize_theta(
    depth_obs: torch.Tensor,
    K: torch.Tensor,
    roi_meta_dict: dict,
    roi_xywh: list,
    rot_tensor: torch.Tensor,
    trans_tensor: torch.Tensor,
    gt_pose_np: np.ndarray,
    gt_beta_np: np.ndarray,
    out_dir: str,
    device: torch.device,
    init_noise_std: float = 0.0,
):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir, "vis_process")
    os.makedirs(vis_dir, exist_ok=True)
    ply_dir = os.path.join(out_dir, "vis_3d")
    os.makedirs(ply_dir, exist_ok=True)

    torch.manual_seed(int(CFG_STAGE2["seed"]))

    skel = SkelAdapter('female', gt_pose_np, gt_beta_np, init_noise_std, device).to(device)
    pose_init = skel.pose.detach().clone()

    if rot_tensor.shape == (3,):
        R = axis_angle_to_matrix(rot_tensor[None])[0]
    else:
        R = rot_tensor
    t = trans_tensor.reshape(1, 3)

    with torch.no_grad():
        skel_gt = SkelAdapter('female', gt_pose_np, gt_beta_np, 0.0, device).to(device)
        V_gt_cam = skel_gt.forward_vertices() @ R.T + t

    faces_t = torch.from_numpy(skel.faces.astype(np.int64)).to(device)
    faces_long = torch.from_numpy(skel.faces.astype(np.int32)).long().to(device)

    optimizer = torch.optim.Adam(
        [{'params': [skel.pose], 'lr': float(CFG_STAGE2["lr_theta"])}]
    )
    iters = int(CFG_STAGE2["iters"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(iters * 0.8)], gamma=0.5)

    print_every = int(CFG_STAGE2["print_every"])
    roi_wh = roi_meta_dict["roi_wh"]

    sigma_space = float(CFG_STAGE2.get("ray_sigma_space", 0.03))
    sigma_z = float(CFG_STAGE2.get("ray_sigma_z", 0.004))  # 4mm（按“米”单位）
    num_pix = int(CFG_STAGE2.get("ray_num_pix", 2048))
    max_ell = int(CFG_STAGE2.get("ray_max_ellipsoids", 5000))
    use_aniso = bool(CFG_STAGE2.get("ray_use_anisotropic", True))
    num_t = int(CFG_STAGE2.get("ray_num_t", 5))

    with torch.no_grad():
        roi_idx = _roi_idx_from_gt(V_gt_cam, K, roi_xywh)
        roi_n = int(roi_idx.numel())
        print(f"[Stage2] ROI eval verts = {roi_n}")

    with torch.no_grad():
        V0_cam = skel.forward_vertices() @ R.T + t
        ellipsoid_log_scales = _init_fixed_ellipsoid_log_scales(V0_cam, faces_t, min_s=0.01)

    print(f"[Stage2] Start Prob-Ray Optim. iters={iters}, Noise={init_noise_std}")
    print(f"[Stage2] Ray params: use_aniso={use_aniso}, sigma_space={sigma_space}, sigma_z={sigma_z}, num_pix={num_pix}, max_ell={max_ell}, num_t={num_t}")

    with torch.no_grad():
        tri0 = V0_cam[faces_t]
        C0 = tri0.mean(dim=1)

        D0_vis = render_gaussian_depth(C0, K, roi_wh, roi_xywh, radius=0.025, sigma_scale=3.0)
        save_depth_vis(depth_obs, os.path.join(vis_dir, "depth_obs.png"))
        save_depth_vis(D0_vis, os.path.join(vis_dir, "depth_pred_start.png"))

        if roi_n >= 20:
            err0 = pa_mpjpe(V0_cam[roi_idx], V_gt_cam[roi_idx]).item()
        else:
            err0 = float("nan")
        print(f"[Stage2][ROI PA-MPJPE] start = {err0:.6f}")

        save_comparison_ply(
            V0_cam.detach().cpu().numpy(),
            V_gt_cam.detach().cpu().numpy(),
            skel.faces,
            os.path.join(ply_dir, "opt_start.ply"),
        )

        save_roi_mesh_ellipsoids_ply(
            v_gt=V_gt_cam,
            v_pred=V0_cam,
            faces=faces_long,
            K=K,
            roi_xywh=roi_xywh,
            centers_pred=C0,
            ellipsoid_log_scales=ellipsoid_log_scales,
            out_ply=os.path.join(ply_dir, "roi_START.ply"),
            max_ellipsoids=int(C0.shape[0]),
            ellipsoid_level=1
        )

        # NEW: 2D overlay (START): ellipsoids + GT human (ROI pixel coords)
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=V_gt_cam,
            centers_cam=C0,
            ellipsoid_log_scales=ellipsoid_log_scales,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(vis_dir, "overlay_ell_gt_start.png"),
            title="Stage2 START: ellipsoids + GT",
            max_gt_points=20000,
            max_ellipsoids=max_ell,
            ellipsoid_level=1.0,
        )

    for it in range(iters):
        optimizer.zero_grad()

        V_cam = skel.forward_vertices() @ R.T + t
        tri = V_cam[faces_t]
        centroids = tri.mean(dim=1)

        loss_depth, stats = ray_overlap_nll(
            centers_cam=centroids,
            K=K,
            roi_xywh=roi_xywh,
            depth_obs=depth_obs,
            sigma_space=sigma_space,
            sigma_z=sigma_z,
            num_pix=num_pix,
            max_ellipsoids=max_ell,
            ellipsoid_log_scales=ellipsoid_log_scales,
            use_anisotropic=use_aniso,
            learn_ellipsoid=False,
            num_t=num_t,
        )

        loss_pose_reg = ((skel.pose - pose_init) ** 2).mean() * float(CFG_STAGE2["theta_l2"])
        loss = loss_depth + loss_pose_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_([skel.pose], max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if it % print_every == 0 or it == iters - 1:
            print(f"[Stage2][{it:03d}/{iters}] L_total={loss.item():.5f} (NLL={loss_depth.item():.5f}, Reg={loss_pose_reg.item():.5f}) "
                  f"[centers={stats['num_centers']} pix={stats['num_pix']} t={stats['num_t']}]")

    with torch.no_grad():
        Vend_cam = skel.forward_vertices() @ R.T + t
        tri_end = Vend_cam[faces_t]
        Cend = tri_end.mean(dim=1)

        Dend_vis = render_gaussian_depth(Cend, K, roi_wh, roi_xywh, radius=0.025, sigma_scale=3.0)
        save_depth_vis(Dend_vis, os.path.join(vis_dir, "depth_pred_end.png"))

        if roi_n >= 20:
            err1 = pa_mpjpe(Vend_cam[roi_idx], V_gt_cam[roi_idx]).item()
        else:
            err1 = float("nan")
        print(f"[Stage2][ROI PA-MPJPE] end   = {err1:.6f}")

        save_comparison_ply(
            Vend_cam.detach().cpu().numpy(),
            V_gt_cam.detach().cpu().numpy(),
            skel.faces,
            os.path.join(ply_dir, "opt_end.ply"),
        )

        save_comparison_ply(
            Vend_cam.detach().cpu().numpy(),
            V0_cam.detach().cpu().numpy(),
            skel.faces,
            os.path.join(ply_dir, "opt_delta.ply"),
        )

        save_roi_mesh_ellipsoids_ply(
            v_gt=V_gt_cam,
            v_pred=Vend_cam,
            faces=faces_long,
            K=K,
            roi_xywh=roi_xywh,
            centers_pred=Cend,
            ellipsoid_log_scales=ellipsoid_log_scales,
            out_ply=os.path.join(ply_dir, "roi_END.ply"),
            max_ellipsoids=int(Cend.shape[0]),
            ellipsoid_level=1
        )

        # NEW: 2D overlay (END): ellipsoids + GT human (ROI pixel coords)
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=V_gt_cam,
            centers_cam=Cend,
            ellipsoid_log_scales=ellipsoid_log_scales,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(vis_dir, "overlay_ell_gt_end.png"),
            title="Stage2 END: ellipsoids + GT",
            max_gt_points=20000,
            max_ellipsoids=max_ell,
            ellipsoid_level=1.0,
        )

    theta_out = {
        'pose': skel.pose.detach().cpu().numpy(),
        'beta': skel.beta.detach().cpu().numpy()
    }
    np.save(os.path.join(out_dir, "theta_opt.npy"), theta_out)
