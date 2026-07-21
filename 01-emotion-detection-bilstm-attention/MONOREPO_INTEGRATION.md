# Monorepo Integration

- Project folder: `01-emotion-detection-bilstm-attention/`
- CI workflow: `.github/workflows/01-emotion-detection-bilstm-attention.yml`
- Streamlit entry point: `01-emotion-detection-bilstm-attention/app/streamlit_app.py`
- Runtime requirements: `01-emotion-detection-bilstm-attention/requirements.txt`
- Project tests are isolated inside the project folder.
- Model training is never executed by CI or Streamlit startup.
- Future numbered projects should replicate this same separation of app, data, models, notebooks, outputs, scripts, source modules, tests, and documentation.
