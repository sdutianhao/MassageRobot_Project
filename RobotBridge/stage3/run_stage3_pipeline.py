import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1.camera import apply_rigid
from stage1.render_depth import render_gaussian_depth
from stage2.skel_adapter import SkelAdapter
from stage2.optim_theta import save_comparison_ply

from utils.vis import save_depth_vis
from utils.ray_likelihood import ray_overlap_nll
from utils.ply_vis import save_roi_mesh_ellipsoids_ply
from utils.metrics import mean_vertex_deviation
from utils.vis_roi import save_overlay_ellipsoids_gt_png

from stage3.gaussian_adapter import GaussianSkinModel


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
    ap.add_argument("--num_pix", type=int, default=2048)
    ap.add_argument("--num_t", type=int, default=5)

    ap.add_argument("--lambda_disp", type=float, default=10.0)
    ap.add_argument("--lambda_shape", type=float, default=0.1)

    ap.add_argument("--no_report_metrics", action="store_true")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    dir_vis_3d = os.path.join(args.out_dir, "vis_3d")
    os.makedirs(dir_vis_3d, exist_ok=True)
    dir_vis = os.path.join(args.out_dir, "vis_process")
    os.makedirs(dir_vis, exist_ok=True)

    s1 = np.load(args.stage1_npz)
    K = torch.from_numpy(s1["K"]).float().to(device)
    rot = torch.from_numpy(s1["rot"]).float().to(device)
    trans = torch.from_numpy(s1["trans"]).float().to(device)
    roi_xywh = s1["roi_xywh"].tolist() if hasattr(s1["roi_xywh"], "tolist") else list(s1["roi_xywh"])

    depth_obs = torch.from_numpy(np.load(args.depth_obs_npy)).float().to(device)
    roi_h, roi_w = int(depth_obs.shape[0]), int(depth_obs.shape[1])
    roi_wh = (roi_w, roi_h)
    save_depth_vis(depth_obs, os.path.join(dir_vis, "depth_obs.png"))

    GT_NPZ = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    gt = np.load(GT_NPZ, allow_pickle=True)
    gt_pose = gt["poses"]
    if gt_pose.shape[1] != 46:
        gt_pose = gt_pose[:, :46] if gt_pose.shape[1] > 46 else np.pad(gt_pose, ((0, 0), (0, 46 - gt_pose.shape[1])))

    skel = SkelAdapter("female", gt_pose, gt["betas"], init_noise_std=0.0, device=device).to(device)
    with torch.no_grad():
        v_gt = skel.forward_vertices().clone()
        if v_gt.dim() == 3 and v_gt.shape[0] == 1:
            v_gt = v_gt[0]

    faces_cpu = skel.faces
    faces_gpu = torch.tensor(skel.faces.astype(np.int32), device=device).long()

    if args.init_noise_std > 0:
        v_init = v_gt + torch.randn_like(v_gt) * float(args.init_noise_std)
    else:
        v_init = v_gt.clone()

    model = GaussianSkinModel(
        init_vertices=v_init,
        faces=faces_gpu,
        K=K,
        rot=rot,
        trans=trans,
        roi_xywh=roi_xywh,
        ellipsoid_s0=float(args.ellipsoid_s0),
        max_ellipsoids=int(args.max_ellipsoids),
        chunk_k=1024,
        device=str(device),
    ).to(device)

    optimizer = optim.Adam([
        {"params": [model.ellipsoid_disp], "lr": float(args.lr)},
        {"params": [model.ellipsoid_shape_raw], "lr": float(args.lr) * 0.2},
    ])

    roi_dev_start = float("nan")
    roi_dev_end = float("nan")

    # ---- START 可视化（椭球中心来自 V'）----
    with torch.no_grad():
        v_start = model().detach()
        v_gt_cam = apply_rigid(v_gt, rot, trans)
        v_start_cam = apply_rigid(v_start, rot, trans)

        save_comparison_ply(
            v_start_cam.detach().cpu().numpy(),
            v_gt_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d, "opt_start.ply"),
        )

        if not args.no_report_metrics:
            roi_dev_start = mean_vertex_deviation(v_start_cam, v_gt_cam, idx=model.roi_v_idx).item()

        c_start_cam = model.ellipsoid_centers_cam(v_start)
        log_s_start = model.ellipsoid_log_scales()

        # Export ALL ROI ellipsoids in visualization
        vis_all_ell = int(model.roi_faces.shape[0])

        save_roi_mesh_ellipsoids_ply(
            v_gt=v_gt_cam,
            v_pred=v_start_cam,
            faces=faces_gpu,
            K=K,
            roi_xywh=roi_xywh,
            centers_pred=c_start_cam,
            ellipsoid_log_scales=log_s_start,
            out_ply=os.path.join(dir_vis_3d, "roi_START.ply"),
            max_ellipsoids=vis_all_ell,
            ellipsoid_level=1,
        )

        # NEW: 2D overlay (START): ellipsoids + GT human
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=v_gt_cam,
            centers_cam=c_start_cam,
            ellipsoid_log_scales=log_s_start,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(dir_vis, "overlay_ell_gt_start.png"),
            title="Stage3 START: ellipsoids + GT",
            max_gt_points=20000,
            max_ellipsoids=None,
            ellipsoid_level=1.0,
        )

        depth_pred_start = render_gaussian_depth(
            c_start_cam, K, roi_wh, roi_xywh, radius=0.025, sigma_scale=3.0
        )
        save_depth_vis(depth_pred_start, os.path.join(dir_vis, "depth_pred_start.png"))

    print(f"[Stage3] iters={int(args.epochs)} | lr={float(args.lr)} | ellipsoids={int(model.roi_faces.shape[0])} | roi_verts={int(model.roi_v_idx.numel())}")
    print(f"[Stage3] ray: sigma_z={float(args.sigma_z)} num_t={int(args.num_t)} num_pix={int(args.num_pix)} max_ell={int(args.max_ellipsoids)} | s0={float(args.ellipsoid_s0)}")
    print(f"[Stage3] reg: lambda_disp={float(args.lambda_disp)} lambda_shape={float(args.lambda_shape)}")

    # ---- 训练：每次先 forward 得到 V'，再从 V' 算 centers_cam 进 NLL ----
    for it in range(1, int(args.epochs) + 1):
        optimizer.zero_grad()

        v_def = model()
        centers_cam = model.ellipsoid_centers_cam(v_def)
        log_scales = model.ellipsoid_log_scales()

        loss_nll, stats = ray_overlap_nll(
            centers_cam=centers_cam,
            K=K,
            roi_xywh=roi_xywh,
            depth_obs=depth_obs,
            sigma_space=float(args.sigma_space),
            sigma_z=float(args.sigma_z),
            num_pix=int(args.num_pix),
            max_ellipsoids=int(args.max_ellipsoids),
            ellipsoid_log_scales=log_scales,
            use_anisotropic=True,
            learn_ellipsoid=True,
            num_t=int(args.num_t),
        )

        loss_disp_reg = (model.ellipsoid_disp ** 2).mean() * float(args.lambda_disp)
        l = model.shape_l()
        loss_shape_reg = (l ** 2).mean() * float(args.lambda_shape)

        loss = loss_nll + loss_disp_reg + loss_shape_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if (it % 50 == 0) or (it == 1) or (it == int(args.epochs)):
            print(
                f"[Stage3][{it:03d}/{int(args.epochs)}] L={loss.item():.5f} (NLL={loss_nll.item():.5f}, disp={loss_disp_reg.item():.5f}, shape={loss_shape_reg.item():.5f}) "
                f"[centers={stats['num_centers']} pix={stats['num_pix']} t={stats['num_t']}]"
            )

    # ---- END 可视化 ----
    with torch.no_grad():
        v_end = model().detach()
        v_end_cam = apply_rigid(v_end, rot, trans)

        save_comparison_ply(
            v_end_cam.detach().cpu().numpy(),
            v_gt_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d, "opt_end.ply"),
        )

        save_comparison_ply(
            v_end_cam.detach().cpu().numpy(),
            v_start_cam.detach().cpu().numpy(),
            faces_cpu,
            os.path.join(dir_vis_3d, "opt_delta.ply"),
        )

        if not args.no_report_metrics:
            roi_dev_end = mean_vertex_deviation(v_end_cam, v_gt_cam, idx=model.roi_v_idx).item()

        c_end_cam = model.ellipsoid_centers_cam(v_end)
        log_s_end = model.ellipsoid_log_scales()

        vis_all_ell = int(model.roi_faces.shape[0])

        save_roi_mesh_ellipsoids_ply(
            v_gt=v_gt_cam,
            v_pred=v_end_cam,
            faces=faces_gpu,
            K=K,
            roi_xywh=roi_xywh,
            centers_pred=c_end_cam,
            ellipsoid_log_scales=log_s_end,
            out_ply=os.path.join(dir_vis_3d, "roi_END.ply"),
            max_ellipsoids=vis_all_ell,
            ellipsoid_level=1,
        )

        # NEW: 2D overlay (END): ellipsoids + GT human
        save_overlay_ellipsoids_gt_png(
            v_gt_cam=v_gt_cam,
            centers_cam=c_end_cam,
            ellipsoid_log_scales=log_s_end,
            K=K,
            roi_xywh=roi_xywh,
            roi_wh=roi_wh,
            out_png=os.path.join(dir_vis, "overlay_ell_gt_end.png"),
            title="Stage3 END: ellipsoids + GT",
            max_gt_points=20000,
            max_ellipsoids=None,
            ellipsoid_level=1.0,
        )

        depth_pred_end = render_gaussian_depth(
            c_end_cam, K, roi_wh, roi_xywh, radius=0.025, sigma_scale=3.0
        )
        save_depth_vis(depth_pred_end, os.path.join(dir_vis, "depth_pred_end.png"))

    np.save(os.path.join(args.out_dir, "displacement_opt.npy"), model.ellipsoid_disp.detach().cpu().numpy())
    np.save(os.path.join(args.out_dir, "shape_opt.npy"), model.ellipsoid_shape_raw.detach().cpu().numpy())
    np.save(os.path.join(args.out_dir, "stage3_meta.npy"), {
        "roi_v_idx": model.roi_v_idx.detach().cpu().numpy(),
        "roi_f_idx": model.roi_f_idx.detach().cpu().numpy(),
        "ellipsoid_s0": float(args.ellipsoid_s0),
    })

    print(f"[*] Done. Results -> {args.out_dir}")
    if not args.no_report_metrics:
        print(f"[Stage3][ROIMeanDev] start={roi_dev_start:.6f}m end={roi_dev_end:.6f}m")


if __name__ == "__main__":
    main()
