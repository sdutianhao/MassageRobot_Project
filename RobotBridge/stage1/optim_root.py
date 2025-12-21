import torch
from .losses import depth_l1, depth_l2


def optimize_root(
    render_fn,
    depth_obs,
    rotvec,
    trans,
    iters,
    lr,
    loss=None,
    loss_type=None,
    trans_reg_w=0.0,
    rot_reg_w=0.0,
    log_every=10
):

    opt = torch.optim.Adam([rotvec, trans], lr=lr)
    logs = []

    mask = depth_obs > 0

    for it in range(iters):
        opt.zero_grad()
        depth_pred = render_fn()

        if loss_type == "l2":
            loss = depth_l2(depth_pred, depth_obs, mask)
        else:
            loss = depth_l1(depth_pred, depth_obs, mask)

        # keep graph alive even if mask is empty
        loss = loss + 0.0 * (rotvec.sum() + trans.sum())


        loss.backward()
        opt.step()

        if it % log_every == 0:
            logs.append({"iter": it, "loss": float(loss.detach())})

    return logs
