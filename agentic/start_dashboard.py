"""Start the Agentic dashboard on a selected port and RUN_ID."""

import os
from pathlib import Path
import runpy
import sys


if len(sys.argv) > 3:
    raise SystemExit("Usage: python start_dashboard.py [PORT] [RUN_ID]")

directory = Path(__file__).resolve().parent
os.chdir(directory)
os.environ["DASHBOARD_PORT"] = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DASHBOARD_PORT", "5000")
if len(sys.argv) > 2:
    os.environ["RUN_ID"] = sys.argv[2]

runpy.run_path(str(directory / "server.py"), run_name="__main__")
