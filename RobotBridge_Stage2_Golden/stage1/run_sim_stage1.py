import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import build_K, apply_rigid, project_points
from stage1.roi import roi_meta, to_roi_pixels
from stage1.render_depth import render_depth_softmin_points
from stage1.optim_root import optimize_root


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG_STAGE1["seed"])



    mesh = trimesh.load(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/mesh/HSMR-ballerina.png.skin_0.obj",
        process=False
    )
    V_full = torch.from_numpy(mesh.vertices).float().to(device)
    V = V_full

    # N = 8000
    # V = torch.randn(N, 3, device=device) * 0.3
    # V[:, 2] += 2.0

    # n_samp = CFG_STAGE1["render"].get("sample_n_verts", None)
    # if n_samp is not None and V.shape[0] > n_samp:
    #     idx = torch.randperm(V.shape[0], device=device)[:n_samp]
    #     V = V[idx]

    n_samp = 2000
    if V.shape[0] > n_samp:
        idx = torch.randperm(V.shape[0], device=device)[:n_samp]
        V = V[idx]

    # ---- camera ----
    Kcfg = CFG_STAGE1["K"]
    K = build_K(**Kcfg, device=device)

    # roi_cfg = CFG_STAGE1["roi_xywh"]
    # roi = roi_meta(CFG_STAGE1["img_wh"], roi_cfg)

    roi_cfg = [0, 0, CFG_STAGE1["img_wh"][0], CFG_STAGE1["img_wh"][1]]
    roi = roi_meta(CFG_STAGE1["img_wh"], roi_cfg)

    # ---- GT root ----
    rot_gt = torch.tensor([0.2, -0.1, 0.05], device=device)
    trans_gt = torch.tensor([0.05, -0.02, 0.0], device=device)

    # ---- render observed depth ----
    with torch.no_grad():
        Vc = apply_rigid(V, rot_gt, trans_gt)
        xy, z = project_points(Vc, K)
        xy_roi = to_roi_pixels(xy, roi_cfg)

        # ✅ 打印调试
        print("num vertices in ROI (GT):", (xy_roi[:, 0] >= 0).sum())

        depth_obs = render_depth_softmin_points(
            xy_roi, z, roi["roi_wh"], **CFG_STAGE1["render"]
        )

    # ---- init params ----
    rot = (rot_gt + 0.20 * torch.randn(3, device=device)).detach().requires_grad_()
    trans = (trans_gt + 0.05 * torch.randn(3, device=device)).detach().requires_grad_()

    def render_pred():
        Vc = apply_rigid(V, rot, trans)
        xy, z = project_points(Vc, K)
        xy_roi = to_roi_pixels(xy, roi_cfg)
        print("num vertices in ROI (Pred):", (xy_roi[:, 0] >= 0).sum())



        return render_depth_softmin_points(
            xy_roi, z, roi["roi_wh"], **CFG_STAGE1["render"]
        )

    out_dir = CFG_STAGE1["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    logs = optimize_root(
        render_fn=render_pred,
        depth_obs=depth_obs,
        rotvec=rot,
        trans=trans,
        **CFG_STAGE1["optim"]
    )

    np.savez(
        os.path.join(out_dir, "stage1_result.npz"),
        rot=rot.detach().cpu().numpy(),
        trans=trans.detach().cpu().numpy(),
        K=K.detach().cpu().numpy(),
        roi_meta=roi,
    )

    # ===== export GT / Pred overlay mesh =====
    # ===== export GT / Pred overlay mesh (valid faces) =====
    V_gt = apply_rigid(V_full, rot_gt, trans_gt).detach().cpu().numpy()
    V_pred = apply_rigid(V_full, rot, trans).detach().cpu().numpy()
    F = mesh.faces.astype(np.int64)

    # merge vertices
    V_all = np.vstack([V_gt, V_pred])

    # merge faces (second mesh faces need offset)
    F_all = np.vstack([F, F + V_gt.shape[0]])

    overlay = trimesh.Trimesh(vertices=V_all, faces=F_all, process=False)

    # per-vertex colors: first part red, second part blue
    C = np.zeros((V_all.shape[0], 4), dtype=np.uint8)
    C[:V_gt.shape[0]] = np.array([255, 0, 0, 120], dtype=np.uint8)
    C[V_gt.shape[0]:] = np.array([0, 0, 255, 120], dtype=np.uint8)
    overlay.visual.vertex_colors = C

    overlay.export(os.path.join(out_dir, "gt_pred_overlay.ply"))

    out_dir = CFG_STAGE1["io"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "depth_obs.npy"), depth_obs.cpu().numpy())
    np.save(os.path.join(out_dir, "depth_pred.npy"), render_pred().detach().cpu().numpy())

    # ---- visualize depth (save png) ----
    depth_obs_np = depth_obs.cpu().numpy()
    depth_pred_np = render_pred().detach().cpu().numpy()

    def save_depth_png(depth, path):
        d = depth.copy()
        d[d <= 0] = np.nan  # mask invalid
        plt.figure(figsize=(5, 5))
        plt.imshow(d, cmap="gray")
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()

    save_depth_png(depth_obs_np, os.path.join(out_dir, "depth_obs.png"))
    save_depth_png(depth_pred_np, os.path.join(out_dir, "depth_pred.png"))

    # ---------- visualization: overlay depth ----------
    depth_obs_np = depth_obs.cpu().numpy()
    depth_pred_np = render_pred().detach().cpu().numpy()

    # normalize for visualization
    def norm(x):
        x = x.copy()
        m = x[x > 0]
        if m.size > 0:
            lo, hi = m.min(), m.max()
            x = (x - lo) / (hi - lo + 1e-6)
        return x

    obs_n = norm(depth_obs_np)
    pred_n = norm(depth_pred_np)

    plt.figure(figsize=(5, 5))
    plt.imshow(obs_n, cmap="Reds", alpha=0.6)
    plt.imshow(pred_n, cmap="Blues", alpha=0.6)
    plt.axis("off")
    plt.title("Depth overlay (Red: obs, Blue: pred)")

    plt.savefig(os.path.join(out_dir, "depth_overlay.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close()

    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(logs, f, indent=2)

    print("[OK] Stage1 simulation finished")
    print("Estimated rot:", rot.detach().cpu().numpy())
    print("Estimated trans:", trans.detach().cpu().numpy())


if __name__ == "__main__":
    main()
