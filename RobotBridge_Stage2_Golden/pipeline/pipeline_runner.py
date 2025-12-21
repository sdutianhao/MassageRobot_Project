import argparse
import os
import sys
import torch
import numpy as np

# 路径适配
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_pipeline(args):
    print(f"==========================================")
    print(f"   Massage Robot 3D Reconstruction Pipeline")
    print(f"   Steps: {args.steps}")
    print(f"   Mock Stage 2: {args.mock_stage2}")
    print(f"==========================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage2_output_dir = os.path.join(args.output_root, 'stage2')
    
    # ---------------------------
    # Stage 2: Parametric Optimization (or Mock)
    # ---------------------------
    if 'stage2' in args.steps:
        print("\n>>> [Pipeline] Entering Stage 2...")
        # (此处省略真实的 Stage 2 调用，如有需要可复原)
        pass

    elif args.mock_stage2:
        print("\n>>> [Pipeline] Stage 2 Skipped. Generating MOCK output...")
        from stage3.mock_generator import MockMeshGenerator
        
        os.makedirs(stage2_output_dir, exist_ok=True)
        generator = MockMeshGenerator(device)
        
        # 生成模拟数据 (标准平面 + 噪声)
        # 注意：这里我们生成一个稍大的平面以匹配 dummy depth 的视场
        verts, faces = generator.generate_plane(res=64, size=0.4)
        verts = generator.apply_perturbation(verts, noise_level=0.002)
        
        # 保存为 Stage 2 输出格式，供 Stage 3 读取
        np.save(os.path.join(stage2_output_dir, "stage2_verts.npy"), verts.cpu().numpy())
        np.save(os.path.join(stage2_output_dir, "stage2_faces.npy"), faces.cpu().numpy())
        print(f"[Pipeline] Mock data saved to {stage2_output_dir}")

    # ---------------------------
    # Stage 3: Gaussian Micro-skin
    # ---------------------------
    if 'stage3' in args.steps:
        print("\n>>> [Pipeline] Entering Stage 3...")
        from stage3.optim_gaussian import run_optimization
        
        # 动态修改参数指向 Stage 2 的输出路径
        args.input_dir = stage2_output_dir
        args.output_dir = os.path.join(args.output_root, 'stage3')
        
        # 确保有 GT Depth (如果是 Mock 模式，我们需要生成一个对应的 dummy GT)
        if args.mock_stage2 and not os.path.exists(args.gt_depth_path):
            print("[Pipeline] Generating Dummy GT Depth for Mock run...")
            # 简单生成一个全零深度图占位，或者你需要更复杂的逻辑
            # 这里仅为了保证代码不崩溃，生成一个 512x512 的零矩阵
            # 在实际调试中，你应该使用 tests/run_simulation.py 那样成对生成的 GT
            dummy_depth = np.zeros((512, 512), dtype=np.float32)
            # 为了让 Loss 不为 0，我们在中心画一个“深度坑”
            dummy_depth[200:300, 200:300] = 0.5 
            os.makedirs(os.path.dirname(args.gt_depth_path), exist_ok=True)
            np.save(args.gt_depth_path, dummy_depth)
            
        run_optimization(args)

    print("\n==========================================")
    print("   Pipeline Finished.")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 流程控制
    parser.add_argument('--steps', nargs='+', default=['stage3'], help='Steps to run')
    parser.add_argument('--mock_stage2', action='store_true', help='Generate mock data instead of running Stage 2')
    
    # 路径参数
    parser.add_argument('--output_root', type=str, default='output', help='Root directory for outputs')
    parser.add_argument('--gt_depth_path', type=str, default='experiments/synthetic_case_001/depth.npy')
    
    # Stage 3 参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_disp', type=float, default=5e-3)
    parser.add_argument('--lr_rot', type=float, default=1e-3)
    parser.add_argument('--lam_disp', type=float, default=10.0)
    parser.add_argument('--lam_smooth', type=float, default=50.0)
    
    args = parser.parse_args()
    
    run_pipeline(args)
