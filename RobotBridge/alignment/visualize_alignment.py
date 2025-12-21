from pathlib import Path
import trimesh


def main():
    case_root = Path(
        "/home/hsmr/MassageRobot_Project/RobotBridge/experiments/synthetic_case_001"
    )

    pc_path = (
        case_root
        / "3dgs"
        / "model"
        / "point_cloud_aligned.ply"
    )

    skin_mesh_path = (
        case_root
        / "mesh"
        / "HSMR-ballerina.png.skin_0.obj"
    )

    assert pc_path.exists(), f"Point cloud not found: {pc_path}"
    assert skin_mesh_path.exists(), f"Skin mesh not found: {skin_mesh_path}"

    # ---------- load assets ----------
    pc = trimesh.load(pc_path, process=False)
    skin = trimesh.load(skin_mesh_path, process=False)

    # ---------- basic sanity check ----------
    print("[Point Cloud]")
    print("  vertices:", pc.vertices.shape)

    print("[Skin Mesh]")
    print("  vertices:", skin.vertices.shape)
    print("  faces   :", skin.faces.shape)

    # ---------- visualization ----------
    # 设置颜色，方便区分
    pc.visual.vertex_colors = [255, 0, 0, 180]      # 红色点云
    skin.visual.face_colors = [200, 200, 200, 80]  # 半透明皮肤

    scene = trimesh.Scene()
    scene.add_geometry(pc)
    scene.add_geometry(skin)

    print("[INFO] Showing scene...")
    scene.show()


if __name__ == "__main__":
    main()
