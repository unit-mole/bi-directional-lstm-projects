"""Launch the Streamlit app from any working directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app" / "streamlit_app.py")],
    cwd=PROJECT_ROOT.parent,
    check=True,
)
