import torch


def roi_meta(img_wh, roi_xywh):
    W, H = img_wh
    x0, y0, w, h = roi_xywh
    assert 0 <= x0 < x0 + w <= W
    assert 0 <= y0 < y0 + h <= H
    return {
        "img_wh": (W, H),
        "roi_xywh": (x0, y0, w, h),
        "roi_wh": (w, h),
    }


def to_roi_pixels(xy_full, roi_xywh):
    x0, y0, _, _ = roi_xywh
    offset = torch.tensor([x0, y0], device=xy_full.device, dtype=xy_full.dtype)
    return xy_full - offset
