import numpy as np
import torch

def load_stage1_result(npz_path, device=None):
    """
    读取标准化的 Stage 1 结果 (Pipeline 协议)
    适配: run_stage2.py 的调用需求
    返回: rot, trans, K, roi_xywh (Tensor/List)
    """
    # print(f"[Stage2] Loading Stage 1 result from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 提取数据并转为 Tensor
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    rot = torch.from_numpy(data["rot"]).to(device=device, dtype=torch.float32)
    trans = torch.from_numpy(data["trans"]).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(data["K"]).to(device=device, dtype=torch.float32)
    
    # 兼容处理 ROI 信息
    if "roi_xywh" in data:
        roi_xywh = data["roi_xywh"]
        if isinstance(roi_xywh, np.ndarray):
            roi_xywh = roi_xywh.tolist()
    else:
        # Fallback: 如果是旧版 npz，尝试从 roi_meta 读
        roi_meta = data["roi_meta"].item()
        roi_xywh = roi_meta.get("roi_xywh", roi_meta.get("xywh"))
        
    return rot, trans, K, roi_xywh
