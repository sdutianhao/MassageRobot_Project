import subprocess
from pathlib import Path
from typing import List, Optional


class BaseCondaRunner:
    """
    在指定 conda 环境和工作目录下，运行 Python 脚本。
    所有具体 Runner（HSMR、3DGS）都继承它。
    """

    def __init__(self, env_name: str, work_dir: Path):
        self.env_name = env_name
        self.work_dir = Path(work_dir).resolve()

    def run_python(self, script_path: str, args: Optional[List[object]] = None) -> int:
        """
        在 env_name 环境中，运行:
            python script_path [args...]
        script_path 是相对 work_dir 的路径。
        """
        script_path = str(script_path)
        cmd = ["conda", "run", "-n", self.env_name, "python", script_path]
        if args:
            cmd.extend([str(a) for a in args])

        print(f"[BaseCondaRunner] cwd={self.work_dir}")
        print(f"[BaseCondaRunner] cmd={' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=self.work_dir, check=True)
        return result.returncode
