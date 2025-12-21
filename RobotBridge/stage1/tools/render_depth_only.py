import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import build_K, apply_rigid, project_points
from stage1.roi import roi_meta, to_roi_pixels
from stage1.render_depth import render_depth_softmin_points


def _save_depth_png(depth: np.ndarray, path: str):
    d = depth.copy()
    d[d <= 0] = np.nan
    plt.figure(figsize=(5, 5))
    plt.imshow(d, cmap="gray")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_obj", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rot", type=float, nargs=3, default=None)     # rotvec
    parser.add_argument("--trans", type=float, nargs=3, default=None)   # xyz
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG_STAGE1["seed"])

    import trimesh
    mesh = trimesh.load(args.mesh_obj, process=False)
    V_full = torch.from_numpy(mesh.vertices).float().to(device)
    V = V_full

    n_samp = CFG_STAGE1["render"].get("sample_n_verts", None)
    if n_samp is not None and V.shape[0] > n_samp:
        idx = torch.randperm(V.shape[0], device=device)[:n_samp]
        V = V[idx]

    # camera / roi
    K = build_K(**CFG_STAGE1["K"], device=device)
    roi_cfg = CFG_STAGE1["roi_xywh"]
    roi = roi_meta(CFG_STAGE1["img_wh"], roi_cfg)

    # pose
    if args.rot is None:
        rot = torch.zeros(3, device=device)
    else:
        rot = torch.tensor(args.rot, device=device, dtype=torch.float32)
    if args.trans is None:
        trans = torch.zeros(3, device=device)
    else:
        trans = torch.tensor(args.trans, device=device, dtype=torch.float32)

    # render depth
    with torch.no_grad():
        Vc = apply_rigid(V, rot, trans)
        xy, z = project_points(Vc, K)
        xy_roi = to_roi_pixels(xy, roi_cfg)
        depth = render_depth_softmin_points(
            xy_roi, z, roi["roi_wh"], **CFG_STAGE1["render"]
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    depth_np = depth.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "depth.npy"), depth_np)
    _save_depth_png(depth_np, os.path.join(out_dir, "depth.png"))

    meta = {
        "K": K.detach().cpu().numpy().tolist(),
        "roi_xywh": list(map(int, roi_cfg)),
        "roi_wh": list(map(int, roi["roi_wh"])),
        "img_wh": list(map(int, CFG_STAGE1["img_wh"])),
        "render_cfg": CFG_STAGE1["render"],
        "pose_rot": rot.detach().cpu().numpy().tolist(),
        "pose_trans": trans.detach().cpu().numpy().tolist(),
        "mesh_obj": args.mesh_obj,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] depth rendered:", os.path.join(out_dir, "depth.npy"))


if __name__ == "__main__":
    main()
