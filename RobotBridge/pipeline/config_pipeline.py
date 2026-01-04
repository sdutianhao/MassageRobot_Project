import os
import numpy as np

# ================= Global Settings =================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 输出根目录 (所有结果必须收敛到这里)
OUT_ROOT = os.path.join(PROJECT_ROOT, "output")

# 默认运行模式
# 1: 仿真闭环 (Syn Data -> Stage1 -> Stage2)
# 2: 跳过对齐 (Syn Data -> Load Pre-calc Stage1 -> Stage2)
# 3: 真实数据 (Real Data -> Stage1 -> Stage2)
MODE = 1

# ================= Data Paths (来自你的真实文件) =================

# 1. 目标 Mesh (使用 Skin Mesh 用于深度渲染)
MESH_OBJ = os.path.join(PROJECT_ROOT, "experiments/synthetic_case_001/mesh/HSMR-ballerina.png.skin_0.obj")

# 2. [Mode 3] 真实深度图路径
REAL_DEPTH_OBS_NPY = os.path.join(PROJECT_ROOT, "experiments/synthetic_case_001/depth/depth_0000.npy")

# 3. [Mode 2] 预设的 Stage 1 结果 (用于跳过 Stage 1 直接跑 Stage 2)
# 我们使用你之前跑出来的 stage1_result.npz，这样格式最兼容
REAL_STAGE1_NPZ = os.path.join(PROJECT_ROOT, "stage1/experiments/stage1_sim_case_001/stage1/stage1_result.npz")

# 4. 相机内参文件 (如果有的话，用于 pipeline 加载)
CAMERA_NPZ = os.path.join(PROJECT_ROOT, "experiments/synthetic_case_001/camera/cam_0000.npz")

# ================= Synthetic GT (Simulation) =================
# 仿真生成深度图时使用的真值
# 后背
DEPTH_SYN_ROT_GT = np.array([0.0, 0.8+3.14, 0.0], dtype=np.float32)
DEPTH_SYN_TRANS_GT = np.array([0.0, 0.35, 0.65], dtype=np.float32)

# DEPTH_SYN_ROT_GT = np.array([0.0, 3.75, 0.0], dtype=np.float32)
# DEPTH_SYN_TRANS_GT = np.array([0.0, -0.25, 0.80], dtype=np.float32)

# ================= Stage 1 Init =================
# 刚体对齐的初始猜测
STAGE1_INIT_ROT = np.array([0.1, 0.0, 0.0], dtype=np.float32)
STAGE1_INIT_TRANS = np.array([0.0, 0.0, 0.4], dtype=np.float32)
