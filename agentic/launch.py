"""启动本目录的脚本或模块，兼容固定 sys.path 的 Windows 便携式 Python。"""

import os
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python launch.py SCRIPT [ARGS] | -m MODULE [ARGS]")
    directory = Path(__file__).resolve().parent
    # 只调整当前进程，避免修改共享 Python 的 ._pth 或混入另一个实验目录。
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    target = sys.argv.pop(1)
    if target == "-m":
        module = sys.argv.pop(1)
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        sys.argv[0] = str(directory / target)
        runpy.run_path(sys.argv[0], run_name="__main__")
