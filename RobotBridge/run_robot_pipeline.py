import sys
from pathlib import Path

from synthetic_case import generate_synthetic_case
from runners.hsmr_runner import HSMRRunner


def get_project_root():
    """
    返回 MassageRobot_Project 根目录的 Path。
    假设当前文件路径为：
        /home/hsmr/MassageRobot_Project/RobotBridge/run_robot_pipeline.py
    则父级的父级就是项目根。
    """
    return Path(__file__).resolve().parents[1]


def run_hsmr(case_root):
    """
    在 hsmr_robot 环境下调用 HSMR4Robot 的 demo + export_mesh，
    并将输出复制到给定的 case_root 目录。
    """
    project_root = get_project_root()
    runner = HSMRRunner(project_root)
    runner.run_full(case_root)


def run_3dgs(case_root: Path):
    print("Running 3DGS for the case ...")
    print(f"3DGS should use data under: {case_root}")

    from runners.gs_runner import GSRunner
    project_root = get_project_root()
    gs_runner = GSRunner(project_root)
    gs_runner.run_full(case_root)



def main():
    project_root = get_project_root()
    bridge_root = project_root / "RobotBridge"
    experiments_root = bridge_root / "experiments"

    print(f"[Info] Project root: {project_root}")
    print(f"[Info] Experiments root: {experiments_root}")

    # 1. 先生成一个伪数据 case，用于测试目录结构和后续流程
    case_root = generate_synthetic_case(experiments_root, case_name="synthetic_case_001")
    print(f"[Info] Synthetic case generated at: {case_root}")

    # 2. 预留的两个步骤，目前仅打印，后面逐步填充
    run_hsmr(case_root)
    run_3dgs(case_root)


if __name__ == "__main__":
    # 保证 RobotBridge 目录在 sys.path 中，便于直接用脚本路径运行
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))

    main()
