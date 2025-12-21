from pathlib import Path
import numpy as np
import trimesh


def inspect_mesh(path: Path, name: str):
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    verts = np.asarray(mesh.vertices)

    v_min = verts.min(axis=0)
    v_max = verts.max(axis=0)
    center = (v_min + v_max) / 2.0
    scale = (v_max - v_min).max()

    print(f"\n[{name}]")
    print(f"  path   : {path}")
    print(f"  min    : {v_min}")
    print(f"  max    : {v_max}")
    print(f"  center : {center}")
    print(f"  scale  : {scale:.6f}")

    return {
        "min": v_min,
        "max": v_max,
        "center": center,
        "scale": scale,
    }


if __name__ == "__main__":
    case_root = Path(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    )

    pc_path = case_root / "3dgs" / "model" / "point_cloud" / "iteration_200" / "point_cloud.ply"
    skin_path = case_root / "mesh" / "HSMR-ballerina.png.skin_0.obj"

    assert pc_path.exists(), f"point cloud not found: {pc_path}"
    assert skin_path.exists(), f"skin mesh not found: {skin_path}"

    pc_info = inspect_mesh(pc_path, "3DGS Point Cloud")
    skin_info = inspect_mesh(skin_path, "HSMR Skin Mesh")
