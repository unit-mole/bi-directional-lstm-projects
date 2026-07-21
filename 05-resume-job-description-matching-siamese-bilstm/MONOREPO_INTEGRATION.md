# Monorepo Integration

Place this folder directly under the root `bi-directional-lstm-projects/` repository. Keep the project-specific GitHub Actions workflow at:

```text
.github/workflows/05-resume-job-description-matching-siamese-bilstm.yml
```

Run commands from the project folder so imports, model paths, and Streamlit data paths remain deterministic.

```bash
cd 05-resume-job-description-matching-siamese-bilstm
python -m pytest -q
python scripts/validate_artifacts.py
streamlit run app/streamlit_app.py
```

The workflow intentionally avoids retraining. Model training is a deliberate local or controlled-compute step, while CI checks source compilation, pure-Python tests, imports, and artifact metadata.
