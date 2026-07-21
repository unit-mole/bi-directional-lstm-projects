from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "app" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
    ]
    raise SystemExit(subprocess.call(command, cwd=project_root))


if __name__ == "__main__":
    main()
