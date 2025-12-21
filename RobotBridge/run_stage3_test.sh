#!/bin/bash

# ================= 配置路径 =================
PROJECT_ROOT="/home/hsmr/MassageRobot_Project/RobotBridge"
PYTHON_EXEC="python"

# 1. GT Mesh (纯净的真值)
GT_MESH="${PROJECT_ROOT}/data/human_model/mesh.obj"

# 2. Stage 1 结果 (用于提供相机参数)
STAGE1_NPZ="${PROJECT_ROOT}/output/20251218_201849_mode1/stage1/stage1_result.npz"

# 3. 目标深度图
DEPTH_OBS="${PROJECT_ROOT}/output/20251218_201849_mode1/source_data/depth_obs.npy"

# 4. 输出目录
OUT_DIR="${PROJECT_ROOT}/output/stage3_noise_test"
# ===========================================

echo "=== Stage 3: Vertex Noise Robustness Test ==="
mkdir -p ${OUT_DIR}

# 运行参数说明:
# --noise_std 0.01: 每个顶点加上 1cm 的随机高斯噪声 (看起来会像表面很脏)
# 预期结果: 优化后的绿色 Mesh 应该比初始的绿色 Mesh 平滑，且贴合蓝色 GT

${PYTHON_EXEC} stage3/run_stage3_pipeline.py \
    --mesh_obj "${GT_MESH}" \
    --stage1_npz "${STAGE1_NPZ}" \
    --depth_obs_npy "${DEPTH_OBS}" \
    --out_dir "${OUT_DIR}" \
    --noise_std 0.01 \
    --lr 0.005 \
    --epochs 200

echo "=== Done ==="
echo "Output saved to: ${OUT_DIR}"
