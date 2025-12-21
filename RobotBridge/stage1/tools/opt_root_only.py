import os
import json
import argparse
import numpy as np
import torch

from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import apply_rigid, project_points
from stage1.roi import to_roi_pixels
from stage1.render_depth import render_depth_softmin_points
from stage1.optim_root import optimize_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_obj", required=True)
    parser.add_argument("--depth_obs_npy", required=True)   # 真实深度 or 你渲染的深度
    parser.add_argument("--meta_json", required=True)       # K / roi 等信息
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG_STAGE1["seed"])

    # load meta
    with open(args.meta_json, "r") as f:
        meta = json.load(f)

    K = torch.tensor(meta["K"], device=device, dtype=torch.float32)
    roi_cfg = meta["roi_xywh"]
    roi_wh = meta["roi_wh"]

    # load depth obs
    depth_obs_np = np.load(args.depth_obs_npy)
    depth_obs = torch.from_numpy(depth_obs_np).to(device=device, dtype=torch.float32)

    # load mesh
    import trimesh
    mesh = trimesh.load(args.mesh_obj, process=False)
    V_full = torch.from_numpy(mesh.vertices).float().to(device)
    V = V_full

    n_samp = CFG_STAGE1["render"].get("sample_n_verts", None)
    if n_samp is not None and V.shape[0] > n_samp:
        idx = torch.randperm(V.shape[0], device=device)[:n_samp]
        V = V[idx]

    # init params (不要用 GT；默认从 0 开始，你也可以之后加扰动版本)
    rot = torch.zeros(3, device=device, requires_grad=True)
    trans = torch.zeros(3, device=device, requires_grad=True)

    def render_pred():
        Vc = apply_rigid(V, rot, trans)
        xy, z = project_points(Vc, K)
        xy_roi = to_roi_pixels(xy, roi_cfg)
        return render_depth_softmin_points(
            xy_roi, z, roi_wh, **CFG_STAGE1["render"]
        )

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    logs = optimize_root(
        render_fn=render_pred,
        depth_obs=depth_obs,
        rotvec=rot,
        trans=trans,
        **CFG_STAGE1["optim"]
    )

    # save result
    np.savez(
        os.path.join(out_dir, "stage1_result.npz"),
        rot=rot.detach().cpu().numpy(),
        trans=trans.detach().cpu().numpy(),
        K=K.detach().cpu().numpy(),
        roi_xywh=np.array(roi_cfg, dtype=np.int64),
        roi_wh=np.array(roi_wh, dtype=np.int64),
        meta_json=args.meta_json,
        mesh_obj=args.mesh_obj,
    )

    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(logs, f, indent=2)

    print("[OK] root optimized:", os.path.join(out_dir, "stage1_result.npz"))
    print("Estimated rot:", rot.detach().cpu().numpy())
    print("Estimated trans:", trans.detach().cpu().numpy())


if __name__ == "__main__":
    main()
