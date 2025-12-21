# stage2/config_stage2.py

CFG_STAGE2 = {
    # 训练
    "seed": 42,
    "lr_theta": 1e-3,
    "iters": 300,
    "print_every": 20,

    # theta 的“调试实现”：先用 per-vertex offset 让链路可跑通（后续替换成真实 SKEL 参数化）
    # theta_dim = 3 * N_verts
    "theta_init_std": 0.0,  # 先从 0 开始，保证稳定
    "theta_l2": 1e-2,       # 强正则，防止单视角病态发散
}
