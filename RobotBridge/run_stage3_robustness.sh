#!/bin/bash
PROJECT_ROOT="/home/hsmr/MassageRobot_Project/RobotBridge"
cd ${PROJECT_ROOT}

# --- 1. 创建打包文件夹 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${PROJECT_ROOT}/output/stage3_test_${TIMESTAMP}"
INPUT_DIR="${RUN_DIR}/inputs"
OUTPUT_DIR="${RUN_DIR}/stage3"

mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}

echo "=== Stage 3 Robustness Test (Run: ${TIMESTAMP}) ==="

# --- 2. 准备数据 ---
FIXED_DATA_DIR="/home/hsmr/MassageRobot_Project/RobotBridge/output/20251218_201849_mode1"
DEPTH_OBS="${FIXED_DATA_DIR}/source_data/depth_obs.npy"

if [ ! -f "$DEPTH_OBS" ]; then
    echo "[Error] Fixed 深度图丢失: $DEPTH_OBS"
    exit 1
fi

STAGE1_NPZ="${INPUT_DIR}/stage1_gt_init.npz"
python -c "
import sys; sys.path.append('${PROJECT_ROOT}')
import numpy as np
import torch
from pipeline.config_pipeline import DEPTH_SYN_ROT_GT, DEPTH_SYN_TRANS_GT
from stage1.config_stage1 import CFG_STAGE1
from stage1.camera import build_K
np.savez('${STAGE1_NPZ}', 
    rot=np.array(DEPTH_SYN_ROT_GT, dtype=np.float32), 
    trans=np.array(DEPTH_SYN_TRANS_GT, dtype=np.float32), 
    K=build_K(**CFG_STAGE1['K'], device=torch.device('cpu')).numpy(), 
    roi_xywh=np.array(CFG_STAGE1['roi_xywh']))
"

# --- 3. 运行 Stage 3 ---
python stage3/run_stage3_pipeline.py \
    --mesh_obj "IGNORED" \
    --stage1_npz "${STAGE1_NPZ}" \
    --depth_obs_npy "${DEPTH_OBS}" \
    --out_dir "${OUTPUT_DIR}" \
    --init_noise_std 0.01 \
    --lr 0.005 \
    --epochs 200

echo "=== 完成 ==="
echo "📁 ${RUN_DIR}"
echo "   ├── stage3/vis_3d/"
echo "   │   ├── opt_start.ply (红=Start, 绿=GT)"
echo "   │   ├── opt_end.ply   (红=End,   绿=GT)"
echo "   │   └── opt_delta.ply (红=Start, 绿=End) <-- [新增对比]"
