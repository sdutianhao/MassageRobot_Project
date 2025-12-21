import numpy as np
import trimesh
from pathlib import Path


def main():
    case_root = Path(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    )

    pc_path = case_root / "3dgs/model/point_cloud_aligned.ply"
    mesh_path = case_root / "mesh/HSMR-ballerina.png.skin_0.obj"
    out_path = case_root / "alignment/gaussian_skin_binding_center.npz"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- load ----------
    pc = trimesh.load(pc_path, process=False)
    mesh = trimesh.load(mesh_path, process=False)

    gaussian_xyz = pc.vertices.astype(np.float32)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    print(f"[INFO] Gaussian points : {gaussian_xyz.shape[0]}")
    print(f"[INFO] Skin vertices   : {vertices.shape[0]}")
    print(f"[INFO] Skin faces      : {faces.shape[0]}")

    # ---------- precompute face centers ----------
    face_centers = vertices[faces].mean(axis=1)  # (F, 3)

    # ---------- closest face query ----------
    prox = trimesh.proximity.ProximityQuery(mesh)
    _, _, face_indices = prox.on_surface(gaussian_xyz)

    bound_centers = face_centers[face_indices]
    offsets = gaussian_xyz - bound_centers

    # ---------- save ----------
    np.savez(
        out_path,
        gaussian_xyz=gaussian_xyz,
        face_index=face_indices,
        face_centers=face_centers,  # (F,3) 真正的每个三角面中心
        bound_centers=bound_centers,  # (N,3) 每个高斯点绑定到的中心
        offset=offsets,
    )

    print(f"[OK] Center-based binding saved to: {out_path}")


if __name__ == "__main__":
    main()
