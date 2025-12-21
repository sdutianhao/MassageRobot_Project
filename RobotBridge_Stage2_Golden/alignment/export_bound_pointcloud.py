import numpy as np
import os

def write_ply_xyz(path, xyz):
    xyz = np.asarray(xyz)
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    case_root = "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    align_dir = os.path.join(case_root, "alignment")

    npz_path = os.path.join(align_dir, "gaussian_skin_binding_center.npz")
    assert os.path.exists(npz_path), f"NPZ not found: {npz_path}"

    data = np.load(npz_path)

    # 必须存在的 key
    required_keys = ["gaussian_xyz", "face_center", "offset"]
    for k in required_keys:
        assert k in data.files, f"Missing key in npz: {k}"

    gaussian_xyz = data["gaussian_xyz"]
    face_center  = data["face_center"]
    offset       = data["offset"]

    # === 一致性断言（失败就直接报错） ===
    assert gaussian_xyz.shape == face_center.shape == offset.shape
    assert gaussian_xyz.shape[1] == 3

    # 验证：gaussian_xyz == face_center + offset
    err = np.linalg.norm((face_center + offset) - gaussian_xyz, axis=1).mean()
    print(f"[CHECK] mean reconstruction error: {err:.6e}")
    assert err < 1e-6, "offset definition mismatch!"

    # === 导出两份点云 ===
    out_centroid = os.path.join(align_dir, "point_cloud_bound_centroid_only.ply")
    out_recon    = os.path.join(align_dir, "point_cloud_reconstructed_from_center_offset.ply")

    write_ply_xyz(out_centroid, face_center)
    write_ply_xyz(out_recon, gaussian_xyz)

    print("[OK] Exported:")
    print(" ", out_centroid)
    print(" ", out_recon)


if __name__ == "__main__":
    main()
