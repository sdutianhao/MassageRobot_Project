import argparse, re
from pathlib import Path

import numpy as np
import torch

from stage3.gmm_likelihood import gmm_surface_nll


def _iter_of(p: Path) -> int:
    m = re.search(r"dbg_it(\d+)\.npz$", p.name)
    return int(m.group(1)) if m else -1


def _pick_dbg(run_dir: Path, which: str) -> Path:
    files = sorted(run_dir.rglob("dbg_it*.npz"), key=_iter_of)
    if not files:
        raise FileNotFoundError(f"no dbg_it*.npz under: {run_dir}")
    return files[0] if which == "start" else files[-1]


def _q(x: torch.Tensor, p: float) -> float:
    return torch.quantile(x, torch.tensor(float(p), device=x.device)).item()


@torch.no_grad()
def _summ(name: str, arr: torch.Tensor):
    if arr.numel() == 0:
        return f"{name}=EMPTY"
    return f"{name}(mean/p95/max)={arr.mean().item():.6f}/{_q(arr,0.95):.6f}/{arr.max().item():.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--stage1_npz", required=True)
    ap.add_argument("--depth_obs_npy", required=True)
    ap.add_argument("--sigma_eval", type=float, default=0.005)   # 用同一个 sigma 评估 start/end，避免被退火混淆
    ap.add_argument("--num_pix", type=int, default=2048)
    ap.add_argument("--max_ell", type=int, default=5000)
    ap.add_argument("--vis_depth_gate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dbg_s = _pick_dbg(run_dir, "start")
    dbg_e = _pick_dbg(run_dir, "end")

    s1 = np.load(args.stage1_npz, allow_pickle=True)
    K = torch.from_numpy(s1["K"]).float().to(args.device)
    roi_xywh = s1["roi_xywh"].tolist() if hasattr(s1["roi_xywh"], "tolist") else list(s1["roi_xywh"])
    depth_obs = torch.from_numpy(np.load(args.depth_obs_npy)).float().to(args.device)

    ds = np.load(dbg_s, allow_pickle=True)
    de = np.load(dbg_e, allow_pickle=True)

    cs = torch.from_numpy(ds["centers_cam"]).float().to(args.device)
    ce = torch.from_numpy(de["centers_cam"]).float().to(args.device)

    ls = torch.from_numpy(ds["log_scales"]).float().to(args.device) if "log_scales" in ds else None
    le = torch.from_numpy(de["log_scales"]).float().to(args.device) if "log_scales" in de else None

    # 用同一随机种子保证像素采样一致
    torch.manual_seed(int(args.seed))
    nll_s, st_s = gmm_surface_nll(
        centers_cam=cs, K=K, roi_xywh=roi_xywh, depth_obs=depth_obs,
        num_pix=int(args.num_pix), max_ellipsoids=int(args.max_ell),
        ellipsoid_log_scales=ls, use_anisotropic=(ls is not None),
        sigma_mult=float(args.sigma_eval), chunk_k=1024, vis_depth_gate=float(args.vis_depth_gate)
    )

    torch.manual_seed(int(args.seed))
    nll_e, st_e = gmm_surface_nll(
        centers_cam=ce, K=K, roi_xywh=roi_xywh, depth_obs=depth_obs,
        num_pix=int(args.num_pix), max_ellipsoids=int(args.max_ell),
        ellipsoid_log_scales=le, use_anisotropic=(le is not None),
        sigma_mult=float(args.sigma_eval), chunk_k=1024, vis_depth_gate=float(args.vis_depth_gate)
    )

    rz_s = torch.from_numpy(ds["rz"]).float().to(args.device) if "rz" in ds else torch.empty((0,), device=args.device)
    rt_s = torch.from_numpy(ds["rt"]).float().to(args.device) if "rt" in ds else torch.empty((0,), device=args.device)
    rz_e = torch.from_numpy(de["rz"]).float().to(args.device) if "rz" in de else torch.empty((0,), device=args.device)
    rt_e = torch.from_numpy(de["rt"]).float().to(args.device) if "rt" in de else torch.empty((0,), device=args.device)

    print(f"[RUN] {run_dir}")
    print(f"[DBG] start={dbg_s.name} end={dbg_e.name}")
    print(f"[EVAL] sigma_eval={args.sigma_eval} num_pix={args.num_pix} seed={args.seed} vis_gate={args.vis_depth_gate}")
    print(f"[NLL ] start={nll_s.item():.6f} (centers={st_s['num_centers']} pix={st_s['num_pix']})")
    print(f"[NLL ] end  ={nll_e.item():.6f} (centers={st_e['num_centers']} pix={st_e['num_pix']})")
    print(f"[DIF ] end-start = {nll_e.item()-nll_s.item():+.6f}  (负数=更贴合 depth_obs)")
    print(f"[RZ  ] {_summ('start', rz_s)} | {_summ('end', rz_e)}")
    print(f"[RT  ] {_summ('start', rt_s)} | {_summ('end', rt_e)}")


if __name__ == "__main__":
    main()
