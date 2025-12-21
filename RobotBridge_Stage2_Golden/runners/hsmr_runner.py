from pathlib import Path
import shutil

from .runner_base import BaseCondaRunner



class HSMRRunner(BaseCondaRunner):
    """
    负责在 hsmr 环境中调用 HSMR4Robot 的脚本。
    当前版本：
        - 直接复用原始 demo 跑一遍
        - 然后用 export_mesh 导出一个 yoga1 的 mesh
    先验证子进程调用链路，后面再接入 case_root。
    """

    def __init__(self, project_root: Path):
        hsmr_root = Path(project_root) / "HSMR4Robot"
        super().__init__(env_name="hsmr_robot", work_dir=hsmr_root)

    def run_demo(self, input_path: str) -> None:

        """
        等价于你之前手动执行的命令：

        python exp/run_demo.py \
            --input_path "data_inputs/demo/example_imgs" \
            --det_bs 4 \
            --rec_bs 16 \
            --max_instances 2
        """
        # 优先尝试 robot 版本脚本，如果不存在则回退到原版
        try:
            script = "exp/run_demo_robot.py"
            self.run_python(
                script,
                [
                    "--input_path", input_path,
                    "--det_bs", 4,
                    "--rec_bs", 16,
                    "--max_instances", 2,
                    "--output_path", str(self.work_dir / "data_outputs" / "demos"),
                ],
            )

        except FileNotFoundError:
            script = "exp/run_demo.py"
            self.run_python(
                script,
                [
                    "--input_path",
                    input_path,
                    "--det_bs",
                    4,
                    "--rec_bs",
                    16,
                    "--max_instances",
                    2,
                ],
            )

    def export_mesh(self) -> None:
        """
        等价于你之前手动执行的命令：

        python exp/misc/export_mesh.py \
            --input_path "data_outputs/demos/HSMR-yoga1.jpeg.npz" \
            --outputs_root "data_outputs/demos/HSMR-yoga1_meshes"
        """
        demos_dir = self.work_dir / "data_outputs" / "demos"
        npz_files = sorted(demos_dir.glob("HSMR-*.npz"))

        if not npz_files:
            raise FileNotFoundError(f"No HSMR npz file found in {demos_dir}")

        demos_dir = self.work_dir / "data_outputs" / "demos"
        npz_files = sorted(demos_dir.glob("HSMR-*.npz"))
        input_npz = str(npz_files[0])
        outputs_root = str(demos_dir / (Path(input_npz).stem.replace(".png", "").replace(".jpeg", "") + "_meshes"))

        try:
            script = "exp/misc/export_mesh_robot.py"
            self.run_python(
                script,
                [
                    "--input_path",
                    input_npz,
                    "--outputs_root",
                    outputs_root,
                ],
            )
        except FileNotFoundError:
            script = "exp/misc/export_mesh.py"
            self.run_python(
                script,
                [
                    "--input_path",
                    input_npz,
                    "--outputs_root",
                    outputs_root,
                ],
            )

    def copy_results_to_case(self, case_root: Path) -> None:
        """
        将 HSMR4Robot 生成的 demo 结果拷贝到当前 case_root 目录下：
            - skeleton: 复制 demo 输出的 npz
            - mesh: 复制对应的 mesh 输出目录到 case_root/mesh 下
        当前策略：自动从 demos 目录中发现 HSMR-*.npz（demo 阶段默认只取第一个）。
        """
        case_root = Path(case_root).resolve()
        skel_dir = case_root / "skeleton"
        mesh_dir = case_root / "mesh"
        skel_dir.mkdir(parents=True, exist_ok=True)
        mesh_dir.mkdir(parents=True, exist_ok=True)

        demos_dir = self.work_dir / "data_outputs" / "demos"

        # ---------- skeleton：自动查找 HSMR 输出的 npz ----------
        npz_files = sorted(demos_dir.glob("HSMR-*.npz"))
        if not npz_files:
            print(f"[HSMRRunner] WARNING: no skeleton npz found in {demos_dir}")
            return

        src_npz = npz_files[0]
        dst_npz = skel_dir / src_npz.name
        print(f"[HSMRRunner] Copy skeleton npz to {dst_npz}")
        shutil.copy2(src_npz, dst_npz)

        # ---------- mesh：直接拷贝 demos 目录下对应的 obj 文件 ----------
        mesh_files = list(demos_dir.glob(f"{src_npz.stem}*.obj"))
        if not mesh_files:
            print(f"[HSMRRunner] WARNING: no mesh obj found for {src_npz.stem} in {demos_dir}")
        else:
            for src_obj in mesh_files:
                dst_obj = mesh_dir / src_obj.name
                print(f"[HSMRRunner] Copy mesh obj to {dst_obj}")
                shutil.copy2(src_obj, dst_obj)


    def run_full(self, case_root: Path) -> None:
        """
        先跑 demo，再导出一次 yoga1 的 mesh，
        然后把结果拷贝到当前 case_root 的 mesh/ 和 skeleton/ 目录下。
        """
        input_dir = str((Path(case_root) / "rgb" / "images").resolve())
        print(f"[HSMRRunner] Running HSMR with input_path = {input_dir}")
        self.run_demo(input_dir)

        print("[HSMRRunner] Exporting mesh ...")
        self.export_mesh()
        print(f"[HSMRRunner] Copying results into case_root = {case_root}")
        self.copy_results_to_case(case_root)
        print("[HSMRRunner] HSMR pipeline finished.")

