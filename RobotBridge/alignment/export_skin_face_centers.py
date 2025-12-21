import numpy as np
import trimesh
from pathlib import Path


def write_ply_xyz(path, xyz):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in xyz:
            f.write(f"{x:.9f} {y:.9f} {z:.9f}\n")


def main():
    case_root = Path(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    )

    mesh_path = case_root / "mesh/HSMR-ballerina.png.skin_0.obj"
    out_ply = case_root / "alignment/skin_face_centers.ply"

    mesh = trimesh.load(mesh_path, process=False)
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)

    print("[INFO] Skin vertices:", vertices.shape[0])
    print("[INFO] Skin faces   :", faces.shape[0])

    face_centers = vertices[faces].mean(axis=1)  # (F,3)

    write_ply_xyz(out_ply, face_centers)

    print("[OK] Saved face centers to:", out_ply)
    print("[OK] Face centers shape  :", face_centers.shape)


if __name__ == "__main__":
    main()
