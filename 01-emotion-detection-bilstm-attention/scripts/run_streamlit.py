from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([sys.executable,"-m","streamlit","run",str(ROOT/"app/streamlit_app.py")]))
