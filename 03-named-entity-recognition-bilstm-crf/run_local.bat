@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run app\streamlit_app.py
if errorlevel 1 (
  echo.
  echo Streamlit failed to start. Install dependencies with:
  echo pip install -r requirements.txt
  exit /b 1
)
