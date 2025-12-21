from pathlib import Path

from .runner_base import BaseCondaRunner


class GSRunner(BaseCondaRunner):
    """
    负责在 3DGS 环境中调用 external/3DGS 的 train.py / render.py。

    说明（当前最小可运行版本）：
        1) 暂时不读取 case_root/rgb，而是直接跑你已经验证过的官方数据 drjohnson；
        2) 目的：先验证 RobotBridge -> (conda隔离) -> 3DGS 的子进程链路可用；
        3) 跑通后再做下一步：把 -s 改为 case_root/rgb 并写入 case_root/3dgs_output。
    """

    def __init__(self, project_root: Path):
        # 先用已跑通的 external/3DGS 作为工作目录（最稳妥）
        gs_root = Path(project_root) / "3DGS4Robot"
        super().__init__(env_name="env_3dgs_robot", work_dir=gs_root)  # 如果你的环境名不同，只改这里

        # 固定一个“已验证可跑”的最小场景，先把链路打通
        self._source_path = None  # 运行时由 case_root 决定：case_root/3dgs/source
        self._model_path = None  # 在 run_full 里根据 case_root 动态设置
        self._iterations = "200"

    def train(self) -> None:
        """
        等价于：
            python train.py -s data/db/drjohnson -m output/db_drjohnson_7000 --iterations 7000
        """
        assert self._model_path is not None, "GSRunner._model_path is not set. Call run_full(case_root) first."

        self.run_python(
            "train.py",
            [
                "-s", self._source_path,
                "-m", self._model_path,
                "--iterations", self._iterations,
                "--disable_viewer",
            ],
        )

    def render(self) -> None:
        """
        等价于：
            python render.py -m output/db_drjohnson_7000
        """
        self.run_python(
            "render.py",
            [
                "-m", self._model_path,
            ],
        )

    def run_full(self, case_root: Path) -> None:
        """
        Bridge 的统一入口（case_root 先作为占位参数保留，后续会用它接入真实数据）。
        """
        print(f"[GSRunner] Running 3DGS pipeline (stub dataset) for case_root = {case_root}")
        self._model_path = str(Path(case_root) / "3dgs" / "model")
        Path(self._model_path).mkdir(parents=True, exist_ok=True)
        source_dir = Path(case_root) / "3dgs" / "source"
        if not source_dir.exists():
            raise FileNotFoundError(f"[GSRunner] 3DGS source dir not found: {source_dir}")
        self._source_path = str(source_dir)

        print("[GSRunner] Step1: train ...")

        case_root = Path(case_root).resolve()
        self._source_path = str(case_root / "3dgs" / "source")
        self._model_path = str(case_root / "3dgs" / "model")

        self.train()

        print("[GSRunner] Step2: render ...")
        self.render()
        print("[GSRunner] 3DGS pipeline finished.")
