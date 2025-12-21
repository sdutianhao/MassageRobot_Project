from pathlib import Path
import numpy as np
import trimesh


def load_mesh(path: Path):
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


if __name__ == "__main__":
    case_root = Path(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    )

    pc_path = case_root / "3dgs" / "model" / "point_cloud" / "iteration_200" / "point_cloud.ply"
    skin_path = case_root / "mesh" / "HSMR-ballerina.png.skin_0.obj"

    pc_mesh = load_mesh(pc_path)
    skin_mesh = load_mesh(skin_path)

    pc_verts = np.asarray(pc_mesh.vertices)
    skin_verts = np.asarray(skin_mesh.vertices)

    # === 已确定的参数（来自你的 inspect 输出） ===
    scale_3dgs = 120.664616
    scale_hsmr = 2.094687

    center_3dgs = np.array([-18.4837997, -27.90050125, 30.33327293])
    center_hsmr = np.array([-0.09584777, -0.28610068, 0.03392571])

    s = scale_hsmr / scale_3dgs
    t = center_hsmr - s * center_3dgs

    print(f"scale = {s}")
    print(f"translation = {t}")

    # === 应用变换 ===
    pc_verts_aligned = pc_verts * s + t

    pc_mesh_aligned = trimesh.points.PointCloud(
        vertices=pc_verts_aligned
    )

    out_path = case_root / "3dgs" / "model" / "point_cloud_aligned.ply"
    pc_mesh_aligned.export(out_path)

    print(f"[OK] Aligned point cloud saved to: {out_path}")
