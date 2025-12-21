import torch

def _procrustes_align_similarity(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
    """
    Similarity Procrustes: find s,R,t s.t. s*pred*R + t best matches gt (L2).
    pred, gt: (N,3)
    return aligned_pred (N,3)
    """
    assert pred.ndim == 2 and gt.ndim == 2 and pred.shape[1] == 3 and gt.shape[1] == 3
    device = pred.device
    pred = pred.float()
    gt = gt.float()

    mu_p = pred.mean(dim=0, keepdim=True)
    mu_g = gt.mean(dim=0, keepdim=True)
    X = pred - mu_p
    Y = gt - mu_g

    # covariance
    H = X.t() @ Y  # (3,3)
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.t() @ U.t()
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.t() @ U.t()

    var_X = (X ** 2).sum()
    s = (S.sum() / (var_X + eps)).clamp(min=eps)

    aligned = (s * (pred - mu_p) @ R) + mu_g
    return aligned

def pa_mpjpe(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred, gt: (N,3) in same units. returns mean Euclidean error after Procrustes alignment.
    """
    aligned = _procrustes_align_similarity(pred, gt, eps=eps)
    err = torch.linalg.norm(aligned - gt, dim=1).mean()
    return err


def mean_vertex_deviation(pred, gt, idx=None):
    """
    Mean Euclidean distance between corresponding vertices.
    pred, gt: (N,3) or (1,N,3) torch.Tensor in the same coordinate frame (meters).
    idx: optional LongTensor/ndarray/list of vertex indices (ROI). If None, use all vertices.
    Returns: scalar torch.Tensor
    """
    import torch

    if pred.dim() == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if gt.dim() == 3 and gt.shape[0] == 1:
        gt = gt[0]

    pred = pred.float()
    gt = gt.float()

    if idx is not None:
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx, device=pred.device)
        idx = idx.long()
        pred = pred[idx]
        gt = gt[idx]

    return torch.linalg.norm(pred - gt, dim=1).mean()
