import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.camera import apply_rigid, project_points
from stage1.render_depth import render_gaussian_depth
from stage2.skel_adapter import SkelAdapter
from stage2.optim_theta import save_comparison_ply

from utils.vis import save_depth_vis
from utils.ray_likelihood import ray_overlap_nll
from utils.gmm_likelihood import gmm_surface_nll
from utils.ply_vis import (
    save_roi_mesh_ellipsoids_ply,
    save_roi_pred_ellipsoids_only_ply,
    save_mesh_only_ply,
    save_roi_pred_mesh_only_ply,
    save_roi_ellipsoids_only_ply,
)
from utils.metrics import mean_vertex_deviation
from utils.vis_roi import save_overlay_ellipsoids_gt_png

from stage3.gaussian_adapter import GaussianSkinModel, GaussianSkinModelVertexGMM


def _unique_edges_from_faces(faces: torch.Tensor) -> torch.Tensor:
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e = torch.sort(e, dim=1).values
    e = torch.unique(e, dim=0)
    return e


def _vertex_normals_from_faces(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)
    vn = torch.zeros_like(V)
    vn.index_add_(0, F[:, 0], fn)
    vn.index_add_(0, F[:, 1], fn)
    vn.index_add_(0, F[:, 2], fn)
    vn = vn / (vn.norm(dim=1, keepdim=True) + 1e-12)
    return vn


def _grad_diffuse_inplace(grad: torch.Tensor, edges: torch.Tensor, alpha: float, every: int, it: int):
    if alpha <= 0.0: return
    if every <= 0: return
    if (it % every) != 0: return
    if grad is None or not isinstance(grad, torch.Tensor) or grad.numel() == 0: return
    if edges is None or not isinstance(edges, torch.Tensor) or edges.numel() == 0: return

    M = int(grad.shape[0])
    src = edges[:, 0]
    dst = edges[:, 1]
    
    if src.max() >= M or dst.max() >= M:
        return

    deg = torch.zeros((M,), device=grad.device, dtype=grad.dtype)
    deg.index_add_(0, src, torch.ones_like(src, dtype=grad.dtype))
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=grad.dtype))

    neigh_sum = torch.zeros_like(grad)
    neigh_sum.index_add_(0, src, grad[dst])
    neigh_sum.index_add_(0, dst, grad[src])

    neigh_mean = neigh_sum / (deg.view(-1, 1) + 1e-12)
    lap = neigh_mean - grad
    grad.add_(lap * float(alpha))


def _anchor_match_pixel(depth_obs: torch.Tensor, u0: int, v0: int, win: int, z_ref: float):
    H, W = int(depth_obs.shape[0]), int(depth_obs.shape[1])
    win = int(max(win, 0))
    u_min, u_max = max(u0 - win, 0), min(u0 + win, W - 1)
    v_min, v_max = max(v0 - win, 0), min(v0 + win, H - 1)

    patch = depth_obs[v_min:v_max + 1, u_min:u_max + 1]
    valid = (patch > 1e-4) & (patch < 50.0)
    if not bool(valid.any().item()):
        return None

    z_ref_t = torch.tensor(float(z_ref), device=depth_obs.device, dtype=depth_obs.dtype)
    diff = torch.abs(patch - z_ref_t)
    diff = torch.where(valid, diff, torch.full_like(diff, 1e9))
    idx = torch.argmin(diff)
    hh = int(idx // diff.shape[1])
    ww = int(idx % diff.shape[1])
    u = u_min + ww
    v = v_min + hh
    z = float(patch[hh, ww].item())
    return int(u), int(v), z


def _backproject(u: torch.Tensor, v: torch.Tensor, z: torch.Tensor, K: torch.Tensor):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / (fx + 1e-12)
    y = (v - cy) * z / (fy + 1e-12)
    return torch.stack([x, y, z], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_obj", default="IGNORED")
    ap.add_argument("--stage1_npz", required=True)
    ap.add_argument("--depth_obs_npy", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_noise_std", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=200)

    ap.add_argument("--ellipsoid_s0", type=float, default=0.01)
    ap.add_argument("--max_ellipsoids", type=int, default=5000)
    ap.add_argument("--sigma_z", type=float, default=0.004)
    ap.add_argument("--sigma_space", type=float, default=0.03)
    ap.add_argument("--num_pix", type=int, default=204800)
    ap.add_argument("--num_t", type=int, default=5)

    ap.add_argument("--lambda_disp", type=float, default=10.0)
    ap.add_argument("--lambda_shape", type=float, default=0.1)
    ap.add_argument("--lambda_tan", type=float, default=0.0)
    ap.add_argument("--lambda_tan_anchor", type=float, default=0.0)
    ap.add_argument("--no_report_metrics", action="store_true")

    ap.add_argument("--data_term", type=str, default="gmm", choices=["ray", "gmm"])
    ap.add_argument("--gmm_sigma_start", type=float, default=0.05)
    ap.add_argument("--gmm_sigma_end", type=float, default=0.005)
    ap.add_argument("--gmm_center_mode", type=str, default="verts", choices=["faces", "verts"])
    ap.add_argument("--update_rot", action="store_true")

    ap.add_argument("--vis_depth_gate", type=float, default=0.06)

    # V2 Args
    ap.add_argument("--lambda_inext", type=float, default=0.0)
    ap.add_argument("--inext_eps_abs", type=float, default=0.002)
    ap.add_argument("--lambda_order", type=float, default=0.0)
    ap.add_argument("--order_margin", type=float, default=0.0)
    ap.add_argument("--lambda_anchor", type=float, default=0.0)
    ap.add_argument("--anchor_k", type=int, default=12)
    ap.add_argument("--anchor_win", type=int, default=5)
    ap.add_argument("--anchor_warmup", type=int, default=50)
    ap.add_argument("--grad_diff_alpha", type=float, default=0.0)
    ap.add_argument("--grad_diff_every", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_every", type=int, default=40)
    ap.add_argument("--debug_dump", action="store_true")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    dir_vis_3d = os.path.join(args.out_dir, "vis_3d")
    os.makedirs(dir_vis_3d, exist_ok=True)
    dir_vis_3d_comp = os.path.join(dir_vis_3d, "composite")
    os.makedirs(dir_vis_3d_comp, exist_ok=True)
    dir_vis = os.path.join(args.out_dir, "vis_process")
    os.makedirs(dir_vis, exist_ok=True)

    # 1. Load Data
    s1 = np.load(args.stage1_npz)
    K = torch.from_numpy(s1["K"]).float().to(device)
    rot = torch.from_numpy(s1["rot"]).float().to(device)
    trans = torch.from_numpy(s1["trans"]).float().to(device)
    roi_xywh = s1["roi_xywh"].tolist() if hasattr(s1["roi_xywh"], "tolist") else list(s1["roi_xywh"])
    roi_wh = (int(s1["roi_xywh"][2]), int(s1["roi_xywh"][3]))

    depth_obs = torch.from_numpy(np.load(args.depth_obs_npy)).float().to(device)
    save_depth_vis(depth_obs, os.path.join(dir_vis, "depth_obs.png"))

    # 2. Init GT
    GT_NPZ = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    gt = np.load(GT_NPZ, allow_pickle=True)
    gt_pose = gt["poses"]
    if gt_pose.shape[1] != 46:
        gt_pose = gt_pose[:, :46] if gt_pose.shape[1] > 46 else np.pad(gt_pose, ((0, 0), (0, 46 - gt_pose.shape[1])))
    skel = SkelAdapter("female", gt_pose, gt["betas"], init_noise_std=0.0, device=device).to(device)
    with torch.no_grad():
        v_gt = skel.forward_vertices().clone()
        if v_gt.dim() == 3 and v_gt.shape[0] == 1: v_gt = v_gt[0]
    
    faces_cpu = skel.faces
    if not torch.is_tensor(faces_cpu):
        faces_cpu = torch.from_numpy(faces_cpu).long() if isinstance(faces_cpu, np.ndarray) else torch.tensor(faces_cpu, dtype=torch.long)
    faces_gpu = faces_cpu.clone().to(device)

    v_init = v_gt.clone()
    if args.init_noise_std > 0:
        v_init += torch.randn_like(v_init) * float(args.init_noise_std)

    # 3. Static Selection
    roi_v_idx_override, roi_f_idx_override = None, None
    runtime_vis_gate = 0.0
    should_run_static_selection = (args.data_term == "gmm") or (float(args.vis_depth_gate) > 0.0)

    if should_run_static_selection:
        selection_gate = float(args.vis_depth_gate) if float(args.vis_depth_gate) > 0.0 else 0.01
        term_label = args.data_term.upper()
        print(f"[Stage3V2][{term_label}] Performing STATIC visibility selection (gate={selection_gate:.4f}m)...")
        with torch.no_grad():
            rx, ry, rw, rh = [float(x) for x in roi_xywh]
            H, W = depth_obs.shape
            
            # Vertices
            v_cam0 = apply_rigid(v_init, rot, trans)
            uv_v, _ = project_points(v_cam0, K)
            z_v = v_cam0[:, 2]
            u_v = torch.round(uv_v[:, 0] - rx).long()
            v_v = torch.round(uv_v[:, 1] - ry).long()
            in_frustum_v = (u_v >= 0) & (u_v < W) & (v_v >= 0) & (v_v < H) & (z_v > 1e-6)
            Dv = torch.zeros_like(z_v)
            Dv[in_frustum_v] = depth_obs[v_v[in_frustum_v], u_v[in_frustum_v]]
            mv = in_frustum_v & (Dv > 1e-4) & (torch.abs(z_v - Dv) <= selection_gate)
            roi_v_idx_override = torch.where(mv)[0].long()

            # Faces
            tri0 = v_init[faces_gpu]
            c0_world = tri0.mean(dim=1)
            c0_cam = apply_rigid(c0_world, rot, trans)
            uv_c, _ = project_points(c0_cam, K)
            z_c = c0_cam[:, 2]
            u_c = torch.round(uv_c[:, 0] - rx).long()
            v_c = torch.round(uv_c[:, 1] - ry).long()
            in_frustum_c = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H) & (z_c > 1e-6)
            Dc = torch.zeros_like(z_c)
            Dc[in_frustum_c] = depth_obs[v_c[in_frustum_c], u_c[in_frustum_c]]
            mc = in_frustum_c & (Dc > 1e-4) & (torch.abs(z_c - Dc) <= selection_gate)
            roi_f_idx_override = torch.where(mc)[0].long()
            
            if roi_f_idx_override.numel() > int(args.max_ellipsoids):
                print(f"[Stage3V2][{term_label}] Subsampling ellipsoids {roi_f_idx_override.numel()} -> {int(args.max_ellipsoids)}")
                sel = torch.randperm(roi_f_idx_override.numel(), device=device)[:int(args.max_ellipsoids)]
                roi_f_idx_override = roi_f_idx_override[sel]
        print(f"[Stage3V2][{term_label}] Fixed Subset: Verts={roi_v_idx_override.numel()}, Faces={roi_f_idx_override.numel()}")
        print(f"[Stage3V2][{term_label}] Runtime dynamic gate DISABLED (set to 0.0).")

    # 4. Model
    if args.data_term == "gmm" and args.gmm_center_mode == "verts":
        model = GaussianSkinModelVertexGMM(v_init, faces_gpu, K, rot, trans, roi_xywh, roi_v_idx_override=roi_v_idx_override).to(device)
    else:
        model = GaussianSkinModel(v_init, faces_gpu, K, rot, trans, roi_xywh, roi_f_idx_override=roi_f_idx_override).to(device)

    roi_dev_start = float("nan")
    roi_dev_end = float("nan")

    # 5. Visualization Start
    with torch.no_grad():
        v_start = model()
        v_gt_cam = apply_rigid(v_gt, rot, trans)
        v_start_cam = apply_rigid(v_start, rot, trans)
        c_start_cam = model.ellipsoid_centers_cam(v_start)
        log_s = model.ellipsoid_log_scales()
        rot_mats = model.ellipsoid_rot_mats_cam()

        if not args.no_report_metrics:
            roi_dev_start = mean_vertex_deviation(v_start_cam, v_gt_cam, idx=model.roi_v_idx).item()

        save_comparison_ply(
            v_start_cam.detach().cpu().numpy(),
            v_gt_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d_comp, "opt_start.ply"),
        )

        save_roi_mesh_ellipsoids_ply(
            v_gt_cam, v_start_cam, faces_gpu, K, roi_xywh, c_start_cam, log_s,
            os.path.join(dir_vis_3d_comp, "roi_START.ply"), ellipsoid_rot_mats=rot_mats
        )
        save_roi_pred_mesh_only_ply(
            v_start_cam, faces_gpu, K, roi_xywh,
            os.path.join(dir_vis_3d, "roi_pred_START.ply")
        )
        save_roi_ellipsoids_only_ply(
            c_start_cam, log_s, K, roi_xywh,
            os.path.join(dir_vis_3d, "roi_ell_START.ply"), ellipsoid_rot_mats=rot_mats
        )
        
        # [Fix]: Add missing human_GT.ply to align with V1
        save_mesh_only_ply(
            v=v_gt_cam,
            faces=faces_gpu,
            out_ply=os.path.join(dir_vis_3d, "human_GT.ply"),
            rgb=(0, 0, 255),
        )
        
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=v_gt_cam,
            centers_cam=c_start_cam,
            ellipsoid_log_scales=log_s,
            ellipsoid_rot_mats=rot_mats,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(dir_vis, "overlay_ell_gt_start.png"),
            title="Stage3V2 START",
            max_gt_points=20000,
            max_ellipsoids=None,
            ellipsoid_level=1.0,
        )

        depth_start = render_gaussian_depth(c_start_cam, K, roi_wh, roi_xywh, radius=0.025)
        save_depth_vis(depth_start, os.path.join(dir_vis, "depth_pred_start.png"))

    # 6. Precompute for V2
    # A) Full edges for Inextensibility
    faces_for_edges = faces_gpu
    edges_full = _unique_edges_from_faces(faces_for_edges)
    
    # B) Local edges for Grad Diffusion
    edges_local = None
    if float(args.grad_diff_alpha) > 0.0 and hasattr(model, "roi_v_idx") and model.roi_v_idx.numel() > 0:
        nV_full = int(v_init.shape[0])
        full2roi = torch.full((nV_full,), -1, device=device, dtype=torch.long)
        full2roi[model.roi_v_idx] = torch.arange(model.roi_v_idx.numel(), device=device)
        e_mapped = full2roi[edges_full]
        valid = (e_mapped[:, 0] >= 0) & (e_mapped[:, 1] >= 0)
        edges_local = e_mapped[valid]

    base_edge_len = None
    if float(args.lambda_inext) > 0.0:
        with torch.no_grad():
            v0 = v_init.clone()
            e0 = edges_full
            base_edge_len = (v0[e0[:, 0]] - v0[e0[:, 1]]).norm(dim=1)

    anchor_roi_idx = None
    if float(args.lambda_anchor) > 0.0 and model.roi_v_idx.numel() > 0:
        k = int(max(1, min(int(args.anchor_k), int(model.roi_v_idx.numel()))))
        step = max(int(model.roi_v_idx.numel() // k), 1)
        anchor_roi_idx = model.roi_v_idx[torch.arange(0, step * k, step, device=device)[:k]]

    # 7. Optim
    opt = optim.Adam(model.parameters(), lr=args.lr)
    iters = int(args.epochs)
    
    print(f"[Stage3V2] Start Optim: iters={iters}, lr={args.lr}")

    # ---- Stats Helpers from V1 (Inlined for compatibility) ----
    def _qstats(x: torch.Tensor):
        x = x.detach().flatten()
        if x.numel() == 0:
            return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
        qs = torch.tensor([0.0, 0.5, 0.9, 0.95, 1.0], device=x.device)
        qv = torch.quantile(x, qs)
        return {
            "mean": float(x.mean().item()),
            "p50": float(qv[1].item()),
            "p90": float(qv[2].item()),
            "p95": float(qv[3].item()),
            "min": float(qv[0].item()),
            "max": float(qv[4].item()),
        }

    def _proj_uv_cam(pts_cam: torch.Tensor, K: torch.Tensor):
        if pts_cam.dim() == 3 and pts_cam.shape[0] == 1: pts_cam = pts_cam[0]
        z = pts_cam[:, 2]
        eps = 1e-12
        xn = pts_cam[:, 0] / (z + eps)
        yn = pts_cam[:, 1] / (z + eps)
        u = K[0,0] * xn + K[0,1] * yn + K[0,2]
        v = K[1,0] * xn + K[1,1] * yn + K[1,2]
        return u, v, z

    def _center_residuals(centers_cam: torch.Tensor, K: torch.Tensor, roi_xywh, depth_obs: torch.Tensor, tau: float):
        rx, ry = float(roi_xywh[0]), float(roi_xywh[1])
        H, W = int(depth_obs.shape[0]), int(depth_obs.shape[1])
        u_full, v_full, z = _proj_uv_cam(centers_cam, K)
        u, v = u_full - rx, v_full - ry
        in_img = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ii = torch.round(u).long().clamp(0, W - 1)
        jj = torch.round(v).long().clamp(0, H - 1)
        D = depth_obs[jj, ii]
        validD = (D > 1e-4) & (D < 50.0)
        m = in_img & validD
        if centers_cam.dim() == 3 and centers_cam.shape[0] == 1: cc = centers_cam[0]
        else: cc = centers_cam
        t = torch.norm(cc, dim=1)
        rz = (z[m] - D[m]).detach()
        rt = (t[m] - D[m]).detach()
        tau_val = float(tau)
        if tau_val <= 0.0:
            gate_z, gate_t = m, m
        else:
            gate_z = m & ((z - D) <= tau_val)
            gate_t = m & ((t - D) <= tau_val)
        return rz, rt, gate_z.detach(), gate_t.detach(), m.detach()

    def _roi_err_stats(v_def: torch.Tensor, v_gt_cam: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, roi_idx: torch.Tensor):
        v_def_cam = apply_rigid(v_def, rot, trans)
        dv = (v_def_cam - v_gt_cam)[roi_idx]
        dn = torch.norm(dv, dim=1)
        dz = dv[:, 2]
        return _qstats(dn), _qstats(dz.abs())

    debug_dir = os.path.join(args.out_dir, "debug")
    debug_csv = os.path.join(debug_dir, "debug_stage3.csv")
    prev_gate_z = None
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        if not os.path.exists(debug_csv):
            with open(debug_csv, "w") as f:
                f.write(",".join(["it","L","NLL","disp","shape","sigma",
                                  "roi_verts","roi_faces","centers_loss",
                                  "centers_inimg","gate_z","gate_t","gate_z_ratio","gate_flips",
                                  "rz_mean","rz_p90","rz_max","rt_mean","rt_p90","rt_max",
                                  "dv_mean","dv_p95","dv_max","dz_abs_mean","dz_abs_p95","dz_abs_max",
                                  "logS_min","logS_max","S_min","S_max",
                                  "sumw_p50","sumw_min","maxw_mean","effk_mean",
                                  "gdisp","gshape"]) + "\n")

    for it in range(1, iters + 1):
        opt.zero_grad()
        
        v_def = model()
        c_cam = model.ellipsoid_centers_cam(v_def)
        log_s = model.ellipsoid_log_scales()
        rot_mats = model.ellipsoid_rot_mats_cam()

        # Data Term
        stats = {}
        if args.data_term == "ray":
            nll, stats = ray_overlap_nll(
                centers_cam=c_cam, K=K, roi_xywh=roi_xywh, depth_obs=depth_obs,
                num_pix=int(args.num_pix), num_t=int(args.num_t),
                sigma_z=float(args.sigma_z), sigma_space=float(args.sigma_space),
                ellipsoid_log_scales=log_s, ellipsoid_rot_mats=rot_mats,
                vis_depth_gate=float(runtime_vis_gate)
            )
        else:
            # Anneal sigma
            progress = (it - 1) / max(iters - 1, 1)
            sigma = math.exp(math.log(args.gmm_sigma_start) * (1 - progress) + math.log(args.gmm_sigma_end) * progress)
            nll, stats = gmm_surface_nll(
                centers_cam=c_cam, K=K, roi_xywh=roi_xywh, depth_obs=depth_obs,
                num_pix=int(args.num_pix), sigma_mult=sigma,
                ellipsoid_log_scales=log_s, ellipsoid_rot_mats=rot_mats,
                vis_depth_gate=float(runtime_vis_gate)
            )
            stats["sigma_mult"] = float(sigma)

        # Reg
        disp = (model.ellipsoid_disp ** 2).mean()
        shape = (model.ellipsoid_shape_raw ** 2).mean()
        
        tan = torch.tensor(0.0, device=device)
        if float(args.lambda_tan) > 0.0 and model.roi_v_idx.numel() > 0:
            vn = _vertex_normals_from_faces(v_def, faces_gpu)
            vid = model.roi_v_idx
            d = model.ellipsoid_disp
            n = vn[vid]
            dn = (d * n).sum(dim=1, keepdim=True) * n
            tan = ((d - dn) ** 2).mean()

        # Inextensibility
        inext = torch.tensor(0.0, device=device)
        if float(args.lambda_inext) > 0.0:
            cur_len = (v_def[edges_full[:, 0]] - v_def[edges_full[:, 1]]).norm(dim=1)
            diff = torch.abs(cur_len - base_edge_len) - float(args.inext_eps_abs)
            inext = (torch.clamp(diff, min=0.0) ** 2).mean()

        # Order
        order = torch.tensor(0.0, device=device)
        if float(args.lambda_order) > 0.0:
            rx, ry = float(roi_xywh[0]), float(roi_xywh[1])
            uv, _ = project_points(c_cam, K)
            zc = c_cam[:, 2]
            u = torch.round(uv[:, 0] - rx).long()
            v = torch.round(uv[:, 1] - ry).long()
            H, W = depth_obs.shape
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (zc > 1e-6)
            z_obs = torch.zeros_like(zc)
            z_obs[in_img] = depth_obs[v[in_img], u[in_img]]
            valid = in_img & (z_obs > 1e-4) & (z_obs < 50.0)
            if valid.any():
                viol = (z_obs[valid] - float(args.order_margin)) - zc[valid]
                order = (torch.clamp(viol, min=0.0) ** 2).mean()

        # Anchor
        anchor = torch.tensor(0.0, device=device)
        if float(args.lambda_anchor) > 0.0 and anchor_roi_idx is not None and it <= int(args.anchor_warmup):
            pts_cam = apply_rigid(v_def[anchor_roi_idx], rot, trans)
            uv, _ = project_points(pts_cam, K)
            z_ref = pts_cam[:, 2]
            u0 = torch.round(uv[:, 0] - float(roi_xywh[0])).long()
            v0 = torch.round(uv[:, 1] - float(roi_xywh[1])).long()
            matched_uvz = []
            valid_indices = []
            for i in range(len(pts_cam)):
                if z_ref[i] <= 1e-6: continue
                m = _anchor_match_pixel(depth_obs, int(u0[i]), int(v0[i]), int(args.anchor_win), float(z_ref[i].item()))
                if m:
                    matched_uvz.append(m)
                    valid_indices.append(i)
            if matched_uvz:
                uu = torch.tensor([m[0] for m in matched_uvz], device=device).float()
                vv = torch.tensor([m[1] for m in matched_uvz], device=device).float()
                zz = torch.tensor([m[2] for m in matched_uvz], device=device).float()
                uu_full = uu + float(roi_xywh[0])
                vv_full = vv + float(roi_xywh[1])
                P_target_cam = _backproject(uu_full, vv_full, zz, K)
                P_src_cam = pts_cam[valid_indices]
                anchor = ((P_src_cam - P_target_cam) ** 2).mean()

        L = (nll + 
             float(args.lambda_disp)*disp + 
             float(args.lambda_shape)*shape + 
             float(args.lambda_tan)*tan + 
             float(args.lambda_inext)*inext + 
             float(args.lambda_order)*order + 
             float(args.lambda_anchor)*anchor)

        L.backward()
        
        # Get gradient stats before stepping
        gdisp_raw = 0.0
        if model.ellipsoid_disp.grad is not None:
             gdisp_raw = model.ellipsoid_disp.grad.norm().item()
        
        # Grad Diffusion
        if float(args.grad_diff_alpha) > 0.0 and model.ellipsoid_disp.grad is not None and edges_local is not None:
             _grad_diffuse_inplace(model.ellipsoid_disp.grad, edges_local, float(args.grad_diff_alpha), int(args.grad_diff_every), it)

        opt.step()

        # ---- Debug Logging (Aligned with V1) ----
        if args.debug and ((it % args.debug_every == 0) or (it == 1) or (it == int(args.epochs))):
            sigma_now = float(stats.get("sigma_mult", 0.0))
            rz, rt, gate_z, gate_t, inimg = _center_residuals(c_cam, K, roi_xywh, depth_obs, float(runtime_vis_gate))
            
            n_inimg = int(inimg.sum().item())
            n_gz = int(gate_z.sum().item())
            n_gt = int(gate_t.sum().item())
            gz_ratio = float(n_gz / max(n_inimg, 1))
            gate_flips = 0
            if prev_gate_z is not None and prev_gate_z.shape == gate_z.shape:
                gate_flips = int((prev_gate_z ^ gate_z).sum().item())
            prev_gate_z = gate_z.clone()

            rz_s = _qstats(rz)
            rt_s = _qstats(rt)
            dv_s, dz_s = _roi_err_stats(v_def, apply_rigid(v_gt, rot, trans), rot, trans, model.roi_v_idx)
            
            ls = log_s.detach().flatten()
            logS_min = float(ls.min().item()) if ls.numel() else float("nan")
            logS_max = float(ls.max().item()) if ls.numel() else float("nan")
            
            gdisp = float(model.ellipsoid_disp.grad.norm().item()) if model.ellipsoid_disp.grad is not None else 0.0
            gshape = float(model.ellipsoid_shape_raw.grad.norm().item()) if model.ellipsoid_shape_raw.grad is not None else 0.0
            
            # Print detailed stats line (aligned with V1)
            print(
                f"[DBG][{it:03d}] inimg={n_inimg} gate_z={n_gz} gate_t={n_gt} flips={gate_flips} | "
                f"rz(mean/p90/max)={rz_s['mean']:.4f}/{rz_s['p90']:.4f}/{rz_s['max']:.4f} "
                f"rt(mean/p90/max)={rt_s['mean']:.4f}/{rt_s['p90']:.4f}/{rt_s['max']:.4f} | "
                f"dv(p95/max)={dv_s['p95']:.4f}/{dv_s['max']:.4f} dz_abs(p95/max)={dz_s['p95']:.4f}/{dz_s['max']:.4f} | "
                f"logS[{logS_min:.2f},{logS_max:.2f}] sum_w(p50/min)=nan/nan effK=nan | "
                f"gdisp_raw={gdisp_raw:.2e} gdisp={gdisp:.2e} gshape={gshape:.2e} tan={float(tan.item()):.3e}"
            )

            with open(debug_csv, "a") as f:
                f.write(",".join(map(str, [
                    it, float(L.item()), float(nll.item()), float(disp.item()), float(shape.item()), sigma_now,
                    int(model.roi_v_idx.numel()), 0, int(stats.get("num_centers", -1)),
                    n_inimg, n_gz, n_gt, gz_ratio, gate_flips,
                    rz_s["mean"], rz_s["p90"], rz_s["max"], rt_s["mean"], rt_s["p90"], rt_s["max"],
                    dv_s["mean"], dv_s["p95"], dv_s["max"], dz_s["mean"], dz_s["p95"], dz_s["max"],
                    logS_min, logS_max, 0, 0, 0, 0, 0, 0, gdisp, gshape
                ])) + "\n")

            if args.debug_dump:
                tag = f"{it:03d}"
                with torch.no_grad():
                    v_now = model()
                    c_now = model.ellipsoid_centers_cam(v_now)
                    v_now_cam = apply_rigid(v_now, rot, trans)
                    save_roi_mesh_ellipsoids_ply(
                        v_gt_cam, v_now_cam, faces_gpu, K, roi_xywh, c_now, log_s,
                        os.path.join(dir_vis_3d_comp, f"roi_{tag}.ply"), ellipsoid_rot_mats=rot_mats
                    )

        if (it % args.debug_every == 0) or (it == 1) or (it == int(args.epochs)):
            # Print summary line (aligned with V1 but with V2 extras)
            sigma_info = f"sigma={stats.get('sigma_mult', 0.0):.4f}"
            print(f"[Stage3V2][{it:03d}/{int(args.epochs)}] L={L.item():.5f} (NLL={nll.item():.5f}, disp={disp.item():.5f}, shape={shape.item():.5f}, tan={tan.item():.3e}, inext={inext.item():.3e}, order={order.item():.3e}, anchor={anchor.item():.3e}) "
                  f"[centers={stats.get('num_centers', -1)} pix={stats.get('num_pix', -1)} {sigma_info}]")

    # 8. Final Vis
    with torch.no_grad():
        v_end = model()
        v_end_cam = apply_rigid(v_end, rot, trans)
        c_end_cam = model.ellipsoid_centers_cam(v_end)
        log_s = model.ellipsoid_log_scales()
        rot_mats = model.ellipsoid_rot_mats_cam()

        if not args.no_report_metrics:
            roi_dev_end = mean_vertex_deviation(v_end_cam, v_gt_cam, idx=model.roi_v_idx).item()

        save_comparison_ply(
            v_end_cam.detach().cpu().numpy(),
            v_gt_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d_comp, "opt_end.ply"),
        )
        save_comparison_ply(
            v_end_cam.detach().cpu().numpy(),
            v_start_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d_comp, "opt_delta.ply"),
        )

        save_roi_mesh_ellipsoids_ply(
            v_gt_cam, v_end_cam, faces_gpu, K, roi_xywh, c_end_cam, log_s,
            os.path.join(dir_vis_3d_comp, "roi_END.ply"), ellipsoid_rot_mats=rot_mats
        )
        save_roi_pred_mesh_only_ply(
            v_end_cam, faces_gpu, K, roi_xywh,
            os.path.join(dir_vis_3d, "roi_pred_END.ply")
        )
        save_roi_ellipsoids_only_ply(
            c_end_cam, log_s, K, roi_xywh,
            os.path.join(dir_vis_3d, "roi_ell_END.ply"), ellipsoid_rot_mats=rot_mats
        )
        
        save_roi_pred_ellipsoids_only_ply(
            v_end_cam, faces_gpu, K, roi_xywh, c_end_cam, log_s,
            os.path.join(dir_vis_3d_comp, "roi_pred_only_END.ply"),
            ellipsoid_rot_mats=rot_mats
        )
        
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=v_gt_cam,
            centers_cam=c_end_cam,
            ellipsoid_log_scales=log_s,
            ellipsoid_rot_mats=rot_mats,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(dir_vis, "overlay_ell_gt_end.png"),
            title="Stage3V2 END",
            max_gt_points=20000,
            max_ellipsoids=None,
            ellipsoid_level=1.0,
        )

        depth_end = render_gaussian_depth(c_end_cam, K, roi_wh, roi_xywh, radius=0.025)
        save_depth_vis(depth_end, os.path.join(dir_vis, "depth_pred_end.png"))

        if not args.no_report_metrics:
            # [Fix]: Change units from mm (*1000) to cm (*100) to align with Stage3 output
            print(f"[Stage3V2][ROIMeanDev] start={roi_dev_start*100:.4f}cm -> {roi_dev_end*100:.4f}cm")

    print(f"[*] Done. Results -> {args.out_dir}")

if __name__ == "__main__":
    main()
