import torch


def depth_l1(pred, obs, mask):
    return torch.mean(torch.abs(pred[mask] - obs[mask]))


def depth_l2(pred, obs, mask):
    return torch.mean((pred[mask] - obs[mask]) ** 2)
