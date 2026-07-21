from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(project_dir / "app" / "streamlit_app.py")],
        cwd=project_dir,
        check=True,
    )


if __name__ == "__main__":
    main()
