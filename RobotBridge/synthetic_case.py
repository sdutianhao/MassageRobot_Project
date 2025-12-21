import numpy as np
from pathlib import Path


def generate_synthetic_case(experiments_root, case_name="synthetic_case_001"):
    """
    在 experiments_root 下生成一个合成样例 case 目录，结构为：

    experiments_root/
        case_name/
            rgb/
            depth/
            camera/
            mesh/
            skeleton/
            micro_skin/
            logs/

    返回值：case_root 的 Path 对象。
    """
    experiments_root = Path(experiments_root).expanduser().resolve()
    case_root = experiments_root / case_name

    subdirs = ["rgb", "depth", "camera", "mesh", "skeleton", "micro_skin", "logs"]
    for name in subdirs:
        (case_root / name).mkdir(parents=True, exist_ok=True)

    # 1. 伪 RGB：用一个全黑图像占位，后面真正接 HSMR 的时候会换成真实 jpg
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    np.save(case_root / "rgb" / "rgb_0000.npy", rgb)

    # 2. 伪深度：一个常数平面 z = 1.0，用来测试深度加载与 loss 计算的流程
    depth = np.ones((256, 256), dtype=np.float32)
    np.save(case_root / "depth" / "depth_0000.npy", depth)

    # 3. 简单相机内参 / 外参，占位用
    K = np.array(
        [
            [500.0, 0.0, 128.0],
            [0.0, 500.0, 128.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32)
    t = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    np.savez(case_root / "camera" / "cam_0000.npz", K=K, R=R, t=t)

    # 4. mesh / skeleton 先放占位文本，后续真正接 HSMR 的输出
    (case_root / "mesh" / "EMPTY_MESH.txt").write_text(
        "TODO: fill with mesh exported from HSMR4Robot."
    )
    (case_root / "skeleton" / "EMPTY_SKEL.txt").write_text(
        "TODO: fill with skeleton parameters exported from HSMR4Robot."
    )

    return case_root
