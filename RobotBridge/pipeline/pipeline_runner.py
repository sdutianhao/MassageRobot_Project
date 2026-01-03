import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
import datetime
import subprocess

# 添加项目路径
sys.path.append("/home/hsmr/MassageRobot_Project/RobotBridge")

# --- 引入配置 ---
from pipeline.config_pipeline import MODE as DEFAULT_MODE
from pipeline.config_pipeline import OUT_ROOT, MESH_OBJ, REAL_DEPTH_OBS_NPY, REAL_STAGE1_NPZ
from pipeline.config_pipeline import DEPTH_SYN_ROT_GT, DEPTH_SYN_TRANS_GT

# --- 引入 Stage 1 模块 ---
from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import build_K, apply_rigid
from stage1.roi import roi_meta
from stage1.render_depth import render_gaussian_depth
from stage1.optim_root import optimize_root

# --- [Stage 2 & Data] 引入参数化模型适配器 (确保同源) ---
from stage2.skel_adapter import SkelAdapter

# --- 引入可视化工具 ---
from utils.vis import save_depth_vis


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ==========================================
# 1. 核心算法模块: 数据生成 (Data)
# ==========================================

def load_real_data(npy_path, vis_dir):
    """[模块: Data] 读取真实深度与内参"""
    print(f"[*] Loading Real Data from: {npy_path}")
    if not os.path.exists(npy_path):
        print(f"[Warn] Real data not found at {npy_path}")
        return None

    data_dir = os.path.dirname(npy_path)
    meta_path = os.path.join(data_dir, "meta.json")
    depth_arr = np.load(npy_path)

    K_cfg = CFG_STAGE1['K']
    roi_cfg = CFG_STAGE1['roi_xywh']
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        K_cfg = meta.get('K', K_cfg)
        roi_cfg = meta.get('roi_xywh', roi_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(K_cfg, dict):
        K = build_K(**K_cfg, device=device)
    else:
        K = torch.tensor(K_cfg, device=device).float()

    depth_tensor = torch.from_numpy(depth_arr).float().to(device)

    if vis_dir:
        _ensure_dir(vis_dir)
        save_depth_vis(depth_tensor, os.path.join(vis_dir, "vis_input_real.png"))

    return {
        'type': 'real',
        'depth_obs': depth_tensor,
        'K': K,
        'roi_xywh': roi_cfg,
        'gt_pose': None
    }


def gen_synthetic_data(mesh_obj_ignored, out_dir):
    """
    [模块: Data] 生成仿真深度图 (Ground Truth)
    核心逻辑: 使用 SkelAdapter 实时生成 Mesh，确保与后续优化目标完全一致。
    """
    # GT 参数路径
    GT_NPZ_PATH = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz"
    print(f"[*] Generating Synthetic Data from PARAMS: {GT_NPZ_PATH}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_dir(out_dir)

    # 1. 加载参数 (Pose & Beta)
    if not os.path.exists(GT_NPZ_PATH):
        raise FileNotFoundError(f"GT Params not found: {GT_NPZ_PATH}")
        
    gt_data = np.load(GT_NPZ_PATH, allow_pickle=True)
    gt_pose = gt_data['poses']
    gt_beta = gt_data['betas']
    
    # 维度对齐
    if gt_pose.shape[1] != 46:
        new_pose = np.zeros((1, 46), dtype=np.float32)
        valid_len = min(gt_pose.shape[1], 46)
        new_pose[:, :valid_len] = gt_pose[:, :valid_len]
        gt_pose = new_pose

    # 2. 实时生成 Mesh
    print("    -> Invoking SkelAdapter to generate consistent mesh...")
    adapter = SkelAdapter('female', gt_pose, gt_beta, init_noise_std=0.0, device=device)
    with torch.no_grad():
        verts = adapter.forward_vertices().cpu().numpy()
        faces = adapter.faces

    # 3. 构建 Trimesh & 采样
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    print(f"    -> Generated Mesh: {mesh.vertices.shape} vertices")
    
    # 高密度采样 (200k点)
    points_surface, _ = trimesh.sample.sample_surface(mesh, 200000)
    V_dense = torch.from_numpy(points_surface).float().to(device)

    # 4. 渲染深度
    K = build_K(**CFG_STAGE1["K"], device=device)
    roi_cfg = CFG_STAGE1["roi_xywh"]
    roi = roi_meta(CFG_STAGE1["img_wh"], roi_cfg)

    rot_gt = torch.tensor(DEPTH_SYN_ROT_GT, device=device).float()
    trans_gt = torch.tensor(DEPTH_SYN_TRANS_GT, device=device).float()

    with torch.no_grad():
        Vc = apply_rigid(V_dense, rot_gt, trans_gt)
        depth_obs = render_gaussian_depth(
            Vc, K, roi['roi_wh'], roi_cfg, radius=0.002
        )
        depth_obs[depth_obs > 50] = 0

    # 5. 保存
    np.save(os.path.join(out_dir, "depth_obs.npy"), depth_obs.cpu().numpy())
    save_depth_vis(depth_obs, os.path.join(out_dir, "vis_depth_obs.png"))

    meta = {
        "K": K.detach().cpu().numpy().tolist(),
        "roi_xywh": list(map(int, roi_cfg)),
        "source_type": "generated_from_skel_params",
        "source_npz": GT_NPZ_PATH
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        'type': 'syn',
        'depth_obs': depth_obs,
        'K': K,
        'roi_xywh': roi_cfg
    }


def generate_gt_init_npz(out_dir):
    """[辅助] 生成 GT 作为 Stage 1 的替代结果 (force_gt_init)"""
    _ensure_dir(out_dir)
    out_npz = os.path.join(out_dir, "stage1_init_gt.npz")

    rot_val = np.array(DEPTH_SYN_ROT_GT, dtype=np.float32)
    trans_val = np.array(DEPTH_SYN_TRANS_GT, dtype=np.float32)

    device = torch.device("cpu")
    roi_xywh = np.array(CFG_STAGE1["roi_xywh"])
    K_val = build_K(**CFG_STAGE1["K"], device=device).numpy()

    np.savez(
        out_npz,
        rot=rot_val,
        trans=trans_val,
        K=K_val,
        roi_xywh=roi_xywh,
        roi_meta={"roi_xywh": roi_xywh.tolist()}
    )
    print(f"[*] Generated GT Init Pose -> {out_npz}")
    return out_npz


def run_stage1(data_ctx, mesh_obj, out_dir):
    """[模块: Stage1] 稀疏顶点优化 (Rigid Alignment)"""
    print(f"[*] Running Stage 1 (Rigid Align) -> {out_dir}")
    device = data_ctx['depth_obs'].device
    _ensure_dir(out_dir)

    from pipeline.config_pipeline import STAGE1_INIT_ROT, STAGE1_INIT_TRANS
    rot = torch.tensor(STAGE1_INIT_ROT, device=device, requires_grad=True).float()
    trans = torch.tensor(STAGE1_INIT_TRANS, device=device, requires_grad=True).float()

    # Stage 1 仍使用 OBJ 加载稀疏点
    mesh = trimesh.load(mesh_obj, process=False)
    V_sparse = torch.from_numpy(mesh.vertices).float().to(device)

    def render_fn():
        Vc = apply_rigid(V_sparse, rot, trans)
        d_pred = render_gaussian_depth(
            Vc, data_ctx['K'],
            roi_meta(CFG_STAGE1["img_wh"], data_ctx['roi_xywh'])['roi_wh'],
            data_ctx['roi_xywh'],
            radius=0.025
        )
        mask = (d_pred < 50) & (data_ctx['depth_obs'] > 0)
        return d_pred, mask

    optimize_root(
        render_fn=render_fn,
        depth_obs=data_ctx['depth_obs'],
        rotvec=rot,
        trans=trans,
        **CFG_STAGE1["optim"]
    )

    out_npz = os.path.join(out_dir, "stage1_result.npz")
    np.savez(
        out_npz,
        rot=rot.detach().cpu().numpy(),
        trans=trans.detach().cpu().numpy(),
        K=data_ctx['K'].detach().cpu().numpy(),
        roi_xywh=np.array(data_ctx['roi_xywh']),
    )
    return out_npz


# ==========================================
# 2. Stage 2 & 3 执行模块 (Subprocess)
# ==========================================

def run_stage2(mesh_obj, stage1_npz, depth_obs_npy, out_dir, init_noise_std=0.0):
    print(f"[*] Running Stage 2 (Optim Theta) -> {out_dir} [Noise={init_noise_std}]")
    _ensure_dir(out_dir)
    cmd = [
        sys.executable, "-m", "stage2.run_stage2",
        "--mesh_obj", mesh_obj,
        "--stage1_npz", stage1_npz,
        "--depth_obs_npy", depth_obs_npy,
        "--out_dir", out_dir,
        "--init_noise_std", str(init_noise_std)
    ]


    subprocess.check_call(cmd)
    return os.path.join(out_dir, "theta_opt.npy")


def run_stage3(mesh_obj, stage1_npz, depth_obs_npy, out_dir, init_noise_std=0.0,
               stage3_data_term="ray", stage3_gmm_sigma_start=0.05, stage3_gmm_sigma_end=0.005, stage3_vis_depth_gate=-1.0, stage3_debug=False, stage3_debug_every=25, stage3_debug_dump=False):
    """
    调用 stage3.run_stage3_pipeline
    """
    print(f"[*] Running Stage 3 (Micro-Skin) -> {out_dir} [Noise={init_noise_std} Term={stage3_data_term}]")
    _ensure_dir(out_dir)
    cmd = [
        sys.executable, "-m", "stage3.run_stage3_pipeline",
        "--mesh_obj", mesh_obj, 
        "--stage1_npz", stage1_npz,
        "--depth_obs_npy", depth_obs_npy,
        "--out_dir", out_dir,
        "--init_noise_std", str(init_noise_std),
        "--lr", "0.005",
        "--epochs", "200",
        "--data_term", str(stage3_data_term),
        "--gmm_sigma_start", str(stage3_gmm_sigma_start),
        "--gmm_sigma_end", str(stage3_gmm_sigma_end),
        "--vis_depth_gate", str(stage3_vis_depth_gate)
    ]
    if bool(stage3_debug):
        cmd += ["--debug", "--debug_every", str(int(stage3_debug_every))]
        if bool(stage3_debug_dump):
            cmd += ["--debug_dump"]
        print("[Stage3][DBG] enabled:", " ".join(cmd))
    subprocess.check_call(cmd)
    return os.path.join(out_dir, "displacement_opt.npy")


# ==========================================
# 3. 主控流程 (Main Pipeline)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RobotBridge Pipeline")
    parser.add_argument('--mode', type=int, default=DEFAULT_MODE)
    parser.add_argument('--steps', nargs='+',
                        default=['data', 'stage1', 'stage2', 'stage3'],
                        choices=['data', 'stage1', 'stage2', 'stage3'])
    parser.add_argument('--use_fixed_data', action='store_true', help="复用之前生成的深度图")
    parser.add_argument('--force_gt_init', action='store_true', help="跳过Stage1优化，使用GT位姿")
    
    # 噪声控制
    parser.add_argument('--perturb_theta', action='store_true', help="[Stage 2] 对初始骨架参数加噪")
    parser.add_argument('--perturb_vertices', action='store_true', help="[Stage 3] 对初始顶点加噪")

    # Stage3 GMM 控制参数
    parser.add_argument('--stage3_data_term', type=str, default='ray', choices=['ray', 'gmm'])
    parser.add_argument('--stage3_gmm_sigma_start', type=float, default=0.05)
    parser.add_argument('--stage3_gmm_sigma_end', type=float, default=0.005)
    parser.add_argument('--stage3_vis_depth_gate', type=float, default=-1.0)
    parser.add_argument('--stage3_debug', action='store_true')
    parser.add_argument('--stage3_debug_every', type=int, default=25)
    parser.add_argument('--stage3_debug_dump', action='store_true')

    args = parser.parse_args()

    # [Hardcode] 固定数据源 (Mode 1 历史数据)
    FIXED_DIR = "/home/hsmr/MassageRobot_Project/RobotBridge/output/20251218_201849_mode1"
    FIXED_DEPTH_NPY = os.path.join(FIXED_DIR, "source_data", "depth_obs.npy")
    FIXED_S1_NPZ = os.path.join(FIXED_DIR, "stage1", "stage1_result.npz")

    # [命名规则修正] 格式: {task}_{timestamp}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if len(args.steps) == 1:
        # 单步骤运行，直接用步骤名作为前缀
        prefix = args.steps[0] 
    elif args.steps == ['data', 'stage1', 'stage2', 'stage3']:
        # 全流程
        prefix = "pipeline"
    else:
        # 混合步骤
        prefix = "_".join(args.steps)
        
    dir_name = f"{prefix}_{timestamp}"
    
    run_dir = _ensure_dir(os.path.join(OUT_ROOT, dir_name))
    print(f"=== Pipeline Start | ID: {dir_name} ===")
    
    # --- [Step 1] Data ---
    data_ctx = None
    depth_obs_path = None

    if args.use_fixed_data:
        print(f"[*] Using FIXED data source: {FIXED_DEPTH_NPY}")
        if not os.path.exists(FIXED_DEPTH_NPY):
            raise FileNotFoundError(f"Fixed data missing at {FIXED_DEPTH_NPY}")
        depth_obs_path = FIXED_DEPTH_NPY
        if 'stage1' in args.steps:
            data_ctx = load_real_data(depth_obs_path, None)
    
    elif 'data' in args.steps:
        # 使用 SkelAdapter 生成
        if args.mode in (1, 2):
            data_ctx = gen_synthetic_data(MESH_OBJ, os.path.join(run_dir, "source_data"))
            depth_obs_path = os.path.join(run_dir, "source_data", "depth_obs.npy")
        elif args.mode == 3:
            data_ctx = load_real_data(REAL_DEPTH_OBS_NPY, os.path.join(run_dir, "source_data"))
            depth_obs_path = REAL_DEPTH_OBS_NPY
    else:
        # Fallback
        if args.mode == 3: depth_obs_path = REAL_DEPTH_OBS_NPY
        else: print("[Warn] No Data step. Assuming source_data exists.")

    # --- [Step 2] Stage 1 ---
    stage1_npz = None

    if args.force_gt_init:
        print("[*] Force GT Init enabled (Generating perfect Stage 1 result)...")
        stage1_npz = generate_gt_init_npz(os.path.join(run_dir, "init_gt"))

    elif 'stage1' in args.steps:
        if data_ctx:
            stage1_npz = run_stage1(data_ctx, MESH_OBJ, os.path.join(run_dir, "stage1"))
        elif args.use_fixed_data:
            print("[!] Skipping Stage 1 compute (Fixed Data mode).")
        else:
            raise RuntimeError("Cannot run Stage 1: Missing Data Context")
    else:
        # 复用
        if args.use_fixed_data and os.path.exists(FIXED_S1_NPZ):
            stage1_npz = FIXED_S1_NPZ
        else:
            stage1_npz = REAL_STAGE1_NPZ

    # --- [Step 3] Stage 2 ---
    if 'stage2' in args.steps:
        if not stage1_npz: raise ValueError("Stage 1 result missing")
        if not depth_obs_path: raise ValueError("Depth obs missing")
        
        noise_theta = 0.02 if args.perturb_theta else 0.0
        run_stage2(MESH_OBJ, stage1_npz, depth_obs_path, os.path.join(run_dir, "stage2"), noise_theta)

    # --- [Step 4] Stage 3 ---
    if 'stage3' in args.steps:
        if not stage1_npz: raise ValueError("Stage 1 result missing")
        if not depth_obs_path: raise ValueError("Depth obs missing")

        noise_vert = 0.01 if args.perturb_vertices else 0.0
        run_stage3(
            MESH_OBJ, stage1_npz, depth_obs_path, os.path.join(run_dir, "stage3"), noise_vert,
            stage3_data_term=args.stage3_data_term,
            stage3_gmm_sigma_start=args.stage3_gmm_sigma_start,
            stage3_gmm_sigma_end=args.stage3_gmm_sigma_end,
            stage3_vis_depth_gate=args.stage3_vis_depth_gate,
            stage3_debug=args.stage3_debug,
            stage3_debug_every=args.stage3_debug_every,
            stage3_debug_dump=args.stage3_debug_dump
        )

    print(f"=== Pipeline Finished | Results: {run_dir} ===")

if __name__ == "__main__":
    main()
