from pathlib import Path
import subprocess
import sys


def main():
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(project_root / "app" / "streamlit_app.py")],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
