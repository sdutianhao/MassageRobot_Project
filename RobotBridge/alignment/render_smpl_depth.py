import os
import json
import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRasterizer, RasterizationSettings,
    PerspectiveCameras
)


def _to_torch(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def render_depth_full_pytorch3d(
    verts_world, faces,
    K, R, t,
    image_size_hw,
    device="cuda",
    blur_radius=0.0,
    faces_per_pixel=1
):
    """
    verts_world: (N,3) 世界/人体坐标系下顶点（你可视作 world）
    faces:      (F,3) 三角面索引（int）
    K:          (3,3) 相机内参
    R:          (3,3) world->cam 旋转
    t:          (3,)  world->cam 平移
    image_size_hw: (H,W)
    返回:
      depth_full: (H,W) float32, 相机坐标系下 Zc（近处小，远处大）
      valid_full: (H,W) bool
    """
    device = torch.device(device)
    H, W = int(image_size_hw[0]), int(image_size_hw[1])

    verts = _to_torch(verts_world, device)  # (N,3)
    faces = torch.as_tensor(faces, device=device, dtype=torch.int64)  # (F,3)

    # PyTorch3D cameras 需要归一化焦距/主点到 NDC。
    # 这里按 PyTorch3D 约定做像素->NDC 的转换：
    #   fx_ndc = 2*fx/W, fy_ndc = 2*fy/H
    #   cx_ndc = (2*cx/W - 1), cy_ndc = (2*cy/H - 1)
    K = _to_torch(K, device)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    focal_length = torch.stack([2.0 * fx / W, 2.0 * fy / H])[None, :]  # (1,2)
    principal_point = torch.stack([2.0 * cx / W - 1.0, 2.0 * cy / H - 1.0])[None, :]  # (1,2)

    R = _to_torch(R, device)[None, :, :]  # (1,3,3)
    t = _to_torch(t, device)[None, :]     # (1,3)

    cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R, T=t,
        in_ndc=True,
        image_size=torch.tensor([[H, W]], device=device, dtype=torch.float32),
        device=device
    )

    mesh = Meshes(verts=[verts], faces=[faces])

    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=blur_radius,
        faces_per_pixel=faces_per_pixel,
        bin_size=None,      # 让 PyTorch3D 自行选择，通常更稳
        max_faces_per_bin=None
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    # fragments.pix_to_face: (1,H,W,K)  K=faces_per_pixel
    # fragments.bary_coords: (1,H,W,K,3)
    pix_to_face = fragments.pix_to_face[0, :, :, 0]     # (H,W)
    bary = fragments.bary_coords[0, :, :, 0, :]         # (H,W,3)

    valid = pix_to_face >= 0

    # 计算相机坐标系下每个顶点 Zc，再对命中的三角形做重心插值，得到真实 Zc 深度
    # world->cam: Xc = R Xw + t
    # 注意：这里 verts_world 就当作 world 坐标
    verts_cam = (verts @ R[0].T) + t[0]  # (N,3)
    z_cam = verts_cam[:, 2]              # (N,)

    # 根据 pix_to_face 找到每个像素命中的 face 三个顶点索引
    faces_v = faces[pix_to_face.clamp(min=0)]  # (H,W,3) 先 clamp，背景会被 mask 掉

    z0 = z_cam[faces_v[:, :, 0]]
    z1 = z_cam[faces_v[:, :, 1]]
    z2 = z_cam[faces_v[:, :, 2]]

    depth = bary[:, :, 0] * z0 + bary[:, :, 1] * z1 + bary[:, :, 2] * z2
    depth[~valid] = float("nan")

    return depth.float().cpu().numpy(), valid.cpu().numpy()


def apply_roi(depth_full, valid_full, roi_bbox=None, roi_mask=None, mode="crop"):
    """
    mode="crop": 输出裁切后的 ROI 深度（形状随 bbox）
    mode="mask": 输出与全图同尺寸，但非 ROI 设为 nan
    roi_bbox: (x1,y1,x2,y2)  x2,y2 为开区间更方便（python slice）
    roi_mask: (H,W) bool
    """
    H, W = depth_full.shape
    if (roi_bbox is None) and (roi_mask is None):
        raise ValueError("roi_bbox 和 roi_mask 至少给一个。")

    if roi_mask is None:
        x1, y1, x2, y2 = map(int, roi_bbox)
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
    else:
        roi_mask = roi_mask.astype(bool)
        ys, xs = np.where(roi_mask)
        if roi_bbox is None and len(xs) > 0:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            roi_bbox = (int(x1), int(y1), int(x2), int(y2))

    valid_roi_full = valid_full & roi_mask

    if mode == "mask":
        depth_roi = depth_full.copy()
        depth_roi[~roi_mask] = np.nan
        valid_roi = valid_roi_full
        return depth_roi, valid_roi, roi_bbox

    if mode == "crop":
        x1, y1, x2, y2 = map(int, roi_bbox)
        depth_roi = depth_full[y1:y2, x1:x2].copy()
        valid_roi = valid_roi_full[y1:y2, x1:x2].copy()
        return depth_roi, valid_roi, roi_bbox

    raise ValueError("mode 只能是 'crop' 或 'mask'。")


def save_depth_package(
    out_dir,
    depth_full, valid_full,
    depth_roi, valid_roi,
    K, R, t,
    image_size_hw,
    roi_bbox,
    roi_mode,
    extra_meta=None
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 保存 npz（深度+mask）
    np.savez_compressed(
        os.path.join(out_dir, "depth_package.npz"),
        depth_full=depth_full.astype(np.float32),
        valid_full=valid_full.astype(np.uint8),
        depth_roi=depth_roi.astype(np.float32),
        valid_roi=valid_roi.astype(np.uint8),
        K=np.asarray(K, dtype=np.float32),
        R=np.asarray(R, dtype=np.float32),
        t=np.asarray(t, dtype=np.float32),
        image_size_hw=np.asarray(image_size_hw, dtype=np.int32),
        roi_bbox=np.asarray(roi_bbox, dtype=np.int32),
        roi_mode=np.string_(roi_mode)
    )

    # 2) 保存 meta.json（给你第二阶段读起来更方便）
    meta = {
        "image_size_hw": [int(image_size_hw[0]), int(image_size_hw[1])],
        "roi_mode": roi_mode,
        "roi_bbox_xyxy": [int(v) for v in roi_bbox],
        "camera": {
            "K": np.asarray(K, dtype=float).tolist(),
            "R_world_to_cam": np.asarray(R, dtype=float).tolist(),
            "t_world_to_cam": np.asarray(t, dtype=float).tolist(),
            "depth_definition": "Zc (camera coordinate forward axis), unit follows t"
        }
    }
    if extra_meta is not None:
        meta.update(extra_meta)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ----------------------------
# 你只需要改下面这个“示例入口”：
# ----------------------------
def main_example():
    """
    你需要替换的内容：
    - verts_world, faces：从你的 SMPL mesh 得到
    - K, R, t：从你的数据或拟合结果得到
    - roi_bbox 或 roi_mask：从你的 ROI 定义得到
    """
    device = "cuda"

    # 示例占位：你要替换为真实数据
    verts_world = np.random.randn(6890, 3).astype(np.float32)
    faces = np.random.randint(0, 6890, size=(13776, 3), dtype=np.int64)

    H, W = 1080, 1920
    K = np.array([[1500.0, 0.0, W / 2],
                  [0.0, 1500.0, H / 2],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0.0, 0.0, 3.0], dtype=np.float32)  # 示例：相机离人体 3 个单位

    # ROI：bbox 示例 (x1,y1,x2,y2)
    roi_bbox = (700, 300, 1200, 800)

    depth_full, valid_full = render_depth_full_pytorch3d(
        verts_world, faces,
        K, R, t,
        image_size_hw=(H, W),
        device=device
    )

    depth_roi, valid_roi, roi_bbox_used = apply_roi(
        depth_full, valid_full,
        roi_bbox=roi_bbox,
        roi_mask=None,
        mode="crop"   # "crop" or "mask"
    )

    save_depth_package(
        out_dir="./depth_out_example",
        depth_full=depth_full,
        valid_full=valid_full,
        depth_roi=depth_roi,
        valid_roi=valid_roi,
        K=K, R=R, t=t,
        image_size_hw=(H, W),
        roi_bbox=roi_bbox_used,
        roi_mode="crop",
        extra_meta={
            "notes": "Replace example data with real SMPL mesh + camera params."
        }
    )


if __name__ == "__main__":
    main_example()
