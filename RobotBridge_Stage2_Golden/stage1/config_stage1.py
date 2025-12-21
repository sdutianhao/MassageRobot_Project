CFG_STAGE1 = {
    "seed": 0,

    # Image / ROI
    "img_wh": (640, 480),
    "roi_xywh": (0, 0, 640, 480),
    "roi_depth_unit": "meter",

    # Camera intrinsics
    "K": {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
    },

    # Rendering
    "render": {
        "sample_n_verts": 600,
        "splat_radius_px": 2,
        "splat_sigma_px": 1.5,
        "softmin_tau": 0.02,
        "near_z": 0.05,
    },

    # Optimization
    "optim": {
        "iters": 200,
        "lr": 5e-2,
        "loss": "l1",
        "trans_reg_w": 1e-4,
        "rot_reg_w": 1e-4,
        "log_every": 10,
    },

    # IO
    "io": {
        "out_dir": "experiments/stage1_sim_case_001/stage1",
        "save_npz": True,
    },
}
