import os
import numpy as np
import torch

from stage1.camera import project_points


def save_stage3_style_comparison_ply(v_gt: torch.Tensor, v_pred: torch.Tensor, faces: torch.Tensor, filename: str):
    """
    stage3 风格对比 ply：
    - gt: 蓝色
    - pred: 绿色
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    vb = v_gt.detach().cpu().numpy()
    vg = v_pred.detach().cpu().numpy()
    f = faces.detach().cpu().numpy().astype(np.int32)
    nv = vb.shape[0]
    nf = f.shape[0]

    with open(filename, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {nv * 2}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {nf * 2}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        for i in range(nv):
            fh.write(f"{vb[i,0]:.6f} {vb[i,1]:.6f} {vb[i,2]:.6f} 0 0 255\n")
        for i in range(nv):
            fh.write(f"{vg[i,0]:.6f} {vg[i,1]:.6f} {vg[i,2]:.6f} 0 255 0\n")
        for i in range(nf):
            fh.write(f"3 {f[i,0]} {f[i,1]} {f[i,2]}\n")
        for i in range(nf):
            fh.write(f"3 {f[i,0] + nv} {f[i,1] + nv} {f[i,2] + nv}\n")

    print(f"[Viz] Saved: {filename}")


def _icosphere(level: int = 1):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=np.float32)
    F = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)

    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    def midpoint(a, b):
        m = (a + b) * 0.5
        return m / (np.linalg.norm(m) + 1e-12)

    for _ in range(level):
        cache = {}
        V_list = V.tolist()

        def mid_idx(i, j):
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            vi = np.array(V_list[key[0]], dtype=np.float32)
            vj = np.array(V_list[key[1]], dtype=np.float32)
            vm = midpoint(vi, vj)
            V_list.append(vm.tolist())
            idx = len(V_list) - 1
            cache[key] = idx
            return idx

        newF = []
        for (i, j, k) in F:
            a = mid_idx(i, j)
            b = mid_idx(j, k)
            c = mid_idx(k, i)
            newF.append([i, a, c])
            newF.append([j, b, a])
            newF.append([k, c, b])
            newF.append([a, b, c])
        V = np.array(V_list, dtype=np.float32)
        F = np.array(newF, dtype=np.int32)

    return V, F


def _roi_mask_vertices(V_cam: torch.Tensor, K: torch.Tensor, roi_xywh):
    rx, ry, rw, rh = [float(x) for x in roi_xywh]
    uv, _ = project_points(V_cam, K)
    u = uv[:, 0]
    v = uv[:, 1]
    z = V_cam[:, 2]
    m = (z > 1e-6) & (u >= rx) & (u < rx + rw) & (v >= ry) & (v < ry + rh)
    return m


def save_roi_overlay_ply(
    v_gt: torch.Tensor,
    v_pred: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    roi_xywh,
    centers_gt: torch.Tensor,
    centers_pred: torch.Tensor,
    out_ply: str,
    sphere_radius: float = 0.06,
    max_spheres: int = 400,
    sphere_level: int = 1
):
    """
    ROI 内叠加：gt(蓝) + pred(绿)，并把“椭球中心”画成球 mesh（黄=gt中心, 红=pred中心）
    sphere_radius 只影响可视化。
    """
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)

    m_gt = _roi_mask_vertices(v_gt, K, roi_xywh)
    m_pr = _roi_mask_vertices(v_pred, K, roi_xywh)

    f = faces.long()
    fg = f[m_gt[f].all(dim=1)]
    fp = f[m_pr[f].all(dim=1)]

    Vg = v_gt.detach().cpu().numpy()
    Vp = v_pred.detach().cpu().numpy()
    fg_np = fg.detach().cpu().numpy().astype(np.int32)
    fp_np = fp.detach().cpu().numpy().astype(np.int32)

    SV, SF = _icosphere(level=sphere_level)
    SV = (SV * float(sphere_radius)).astype(np.float32)
    SF = SF.astype(np.int32)

    cg = centers_gt.detach()
    cp = centers_pred.detach()
    mgc = _roi_mask_vertices(cg, K, roi_xywh)
    mpc = _roi_mask_vertices(cp, K, roi_xywh)
    cg = cg[mgc]
    cp = cp[mpc]

    def subsample(C):
        if C.shape[0] <= max_spheres:
            return C
        idx = torch.linspace(0, C.shape[0]-1, steps=max_spheres).long().to(C.device)
        return C[idx]

    cg = subsample(cg).cpu().numpy()
    cp = subsample(cp).cpu().numpy()

    verts = []
    colors = []
    faces_out = []

    offset_gt = 0
    verts.append(Vg)
    colors.append(np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (Vg.shape[0], 1)))

    offset_pr = offset_gt + Vg.shape[0]
    verts.append(Vp)
    colors.append(np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (Vp.shape[0], 1)))

    for tri in fg_np:
        faces_out.append([tri[0] + offset_gt, tri[1] + offset_gt, tri[2] + offset_gt])
    for tri in fp_np:
        faces_out.append([tri[0] + offset_pr, tri[1] + offset_pr, tri[2] + offset_pr])

    offset_sg = offset_pr + Vp.shape[0]
    for c in cg:
        v_s = SV + c[None, :]
        verts.append(v_s)
        colors.append(np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (v_s.shape[0], 1)))
        for tri in SF:
            faces_out.append([tri[0] + offset_sg, tri[1] + offset_sg, tri[2] + offset_sg])
        offset_sg += v_s.shape[0]

    offset_sp = offset_sg
    for c in cp:
        v_s = SV + c[None, :]
        verts.append(v_s)
        colors.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (v_s.shape[0], 1)))
        for tri in SF:
            faces_out.append([tri[0] + offset_sp, tri[1] + offset_sp, tri[2] + offset_sp])
        offset_sp += v_s.shape[0]

    V_all = np.concatenate(verts, axis=0).astype(np.float32)
    C_all = np.concatenate(colors, axis=0).astype(np.uint8)
    F_all = np.array(faces_out, dtype=np.int32)

    with open(out_ply, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {V_all.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {F_all.shape[0]}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for i in range(V_all.shape[0]):
            fh.write(f"{V_all[i,0]:.6f} {V_all[i,1]:.6f} {V_all[i,2]:.6f} {int(C_all[i,0])} {int(C_all[i,1])} {int(C_all[i,2])}\n")
        for i in range(F_all.shape[0]):
            fh.write(f"3 {F_all[i,0]} {F_all[i,1]} {F_all[i,2]}\n")

    print(f"[Viz] Saved: {out_ply}")


def save_roi_mesh_ellipsoids_ply(
    v_gt: torch.Tensor,
    v_pred: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    roi_xywh,
    centers_pred: torch.Tensor,
    ellipsoid_log_scales: torch.Tensor,
    out_ply: str,
    max_ellipsoids: int = 400,
    ellipsoid_level: int = 1
):
    """
    ROI 对比：gt(蓝) + pred(绿) + ROI 椭球(红，来自 pred 的 centers + log_scales)。
    注意：只保留 ROI 内的 faces；椭球也按 ROI 过滤并最多保留 max_ellipsoids 个。
    """
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)

    m_gt = _roi_mask_vertices(v_gt, K, roi_xywh)
    m_pr = _roi_mask_vertices(v_pred, K, roi_xywh)

    f = faces.long()
    fg = f[m_gt[f].all(dim=1)]
    fp = f[m_pr[f].all(dim=1)]

    Vg = v_gt.detach().cpu().numpy()
    Vp = v_pred.detach().cpu().numpy()
    fg_np = fg.detach().cpu().numpy().astype(np.int32)
    fp_np = fp.detach().cpu().numpy().astype(np.int32)

    # 椭球（来自 pred）
    cp = centers_pred.detach()
    sp = ellipsoid_log_scales.detach()

    mpc = _roi_mask_vertices(cp, K, roi_xywh)
    cp = cp[mpc]
    sp = sp[mpc]

    def subsample_pair(C, S):
        n = int(C.shape[0])
        if n <= max_ellipsoids:
            return C, S
        idx = torch.linspace(0, n - 1, steps=max_ellipsoids).long().to(C.device)
        return C[idx], S[idx]

    cp, sp = subsample_pair(cp, sp)
    cp_np = cp.cpu().numpy()
    scales_np = torch.exp(sp).cpu().numpy()  # (M,3)

    EV, EF = _icosphere(level=ellipsoid_level)
    EV = EV.astype(np.float32)
    EF = EF.astype(np.int32)

    verts = []
    colors = []
    faces_out = []

    # gt mesh (blue)
    offset_gt = 0
    verts.append(Vg)
    colors.append(np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (Vg.shape[0], 1)))

    # pred mesh (green)
    offset_pr = offset_gt + Vg.shape[0]
    verts.append(Vp)
    colors.append(np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (Vp.shape[0], 1)))

    for tri in fg_np:
        faces_out.append([tri[0] + offset_gt, tri[1] + offset_gt, tri[2] + offset_gt])
    for tri in fp_np:
        faces_out.append([tri[0] + offset_pr, tri[1] + offset_pr, tri[2] + offset_pr])

    # ellipsoids (red)
    offset_e = offset_pr + Vp.shape[0]
    for c, s in zip(cp_np, scales_np):
        v_e = EV * s[None, :] + c[None, :]
        verts.append(v_e)
        colors.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (v_e.shape[0], 1)))
        for tri in EF:
            faces_out.append([tri[0] + offset_e, tri[1] + offset_e, tri[2] + offset_e])
        offset_e += v_e.shape[0]

    V_all = np.concatenate(verts, axis=0).astype(np.float32)
    C_all = np.concatenate(colors, axis=0).astype(np.uint8)
    F_all = np.array(faces_out, dtype=np.int32)

    with open(out_ply, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {V_all.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {F_all.shape[0]}\n")
        fh.write("property list uchar int vertex_indices\n")
        fh.write("end_header\n")
        for i in range(V_all.shape[0]):
            fh.write(f"{V_all[i,0]:.6f} {V_all[i,1]:.6f} {V_all[i,2]:.6f} {int(C_all[i,0])} {int(C_all[i,1])} {int(C_all[i,2])}\n")
        for i in range(F_all.shape[0]):
            fh.write(f"3 {F_all[i,0]} {F_all[i,1]} {F_all[i,2]}\n")

    print(f"[Viz] Saved: {out_ply}")

def save_roi_pred_ellipsoids_only_ply(
    v_pred: torch.Tensor,
    faces: torch.Tensor,
    K: torch.Tensor,
    roi_xywh,
    centers_pred: torch.Tensor,
    ellipsoid_log_scales: torch.Tensor,
    out_ply: str,
    max_ellipsoids: int = 400,
    ellipsoid_level: int = 1
):
    """
    仅保存 ROI 内的预测结果：pred(绿) + ROI 椭球(红)。不包含 GT。
    """
    os.makedirs(os.path.dirname(out_ply), exist_ok=True)
    m_pr = _roi_mask_vertices(v_pred, K, roi_xywh)
    f = faces.long()
    fp = f[m_pr[f].all(dim=1)]
    Vp = v_pred.detach().cpu().numpy()
    fp_np = fp.detach().cpu().numpy().astype(np.int32)
    cp = centers_pred.detach()
    sp = ellipsoid_log_scales.detach()
    mpc = _roi_mask_vertices(cp, K, roi_xywh)
    cp = cp[mpc]
    sp = sp[mpc]
    def subsample_pair(C, S):
        n = int(C.shape[0])
        if n <= max_ellipsoids: return C, S
        idx = torch.linspace(0, n - 1, steps=max_ellipsoids).long().to(C.device)
        return C[idx], S[idx]
    cp, sp = subsample_pair(cp, sp)
    cp_np = cp.cpu().numpy()
    scales_np = torch.exp(sp).cpu().numpy()
    EV, EF = _icosphere(level=ellipsoid_level)
    EV = EV.astype(np.float32)
    EF = EF.astype(np.int32)
    verts, colors, faces_out = [], [], []
    offset_pr = 0
    verts.append(Vp)
    colors.append(np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (Vp.shape[0], 1)))
    for tri in fp_np:
        faces_out.append([tri[0] + offset_pr, tri[1] + offset_pr, tri[2] + offset_pr])
    offset_e = offset_pr + Vp.shape[0]
    for c, s in zip(cp_np, scales_np):
        v_e = EV * s[None, :] + c[None, :]
        verts.append(v_e)
        colors.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (v_e.shape[0], 1)))
        for tri in EF:
            faces_out.append([tri[0] + offset_e, tri[1] + offset_e, tri[2] + offset_e])
        offset_e += v_e.shape[0]
    V_all = np.concatenate(verts, axis=0).astype(np.float32)
    C_all = np.concatenate(colors, axis=0).astype(np.uint8)
    F_all = np.array(faces_out, dtype=np.int32)
    with open(out_ply, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {V_all.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write(f"element face {F_all.shape[0]}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        for i in range(V_all.shape[0]):
            fh.write(f"{V_all[i,0]:.6f} {V_all[i,1]:.6f} {V_all[i,2]:.6f} {int(C_all[i,0])} {int(C_all[i,1])} {int(C_all[i,2])}\n")
        for i in range(F_all.shape[0]):
            fh.write(f"3 {F_all[i,0]} {F_all[i,1]} {F_all[i,2]}\n")
    print(f"[Viz] Saved Pred-Only ROI: {out_ply}")
