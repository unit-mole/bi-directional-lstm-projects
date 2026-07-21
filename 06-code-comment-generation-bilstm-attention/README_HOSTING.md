# Hosting Guide

## Recommended option: Streamlit Community Cloud

### Before deployment

1. Retrain the corrected attention model or explicitly keep the legacy-checkpoint warning visible.
2. Confirm these files are committed:
   - `app/streamlit_app.py`
   - `requirements.txt`
   - `.streamlit/config.toml` or the monorepo root `.streamlit/config.toml`
   - `models/model_metadata.json`
   - tokenizer JSON files
   - required `.keras` artifacts through Git LFS
3. Never commit `.streamlit/secrets.toml`.

### Git and Git LFS

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add .
git commit -m "Complete code comment generation BiLSTM project"
git push origin main
```

### Deploy

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app and select `bi-directional-lstm-projects`.
3. Choose the branch, normally `main`.
4. Set the entrypoint to:

```text
06-code-comment-generation-bilstm-attention/app/streamlit_app.py
```

5. Open advanced settings and select Python 3.11 when available.
6. Deploy, review the build log, and test every safe sample.
7. Add the resulting `*.streamlit.app` URL to the project README, root README, resume, LinkedIn, and portfolio.

Streamlit Community Cloud installs declared dependencies from a dependency file and supports apps whose entrypoint is inside a subdirectory. Keep one authoritative dependency file close to the entrypoint and one repository-level Streamlit configuration for the deployed branch.

## Hugging Face Spaces alternative

1. Create a new **Streamlit** or **Docker** Space.
2. Push this project as the Space repository or use the supplied Dockerfile.
3. Include the model through Git LFS.
4. For Docker Spaces, expose port `7860` by changing the Docker command to run Streamlit on that port.
5. Keep the responsible-use warning and code privacy notice visible.

## Deployment checklist

- App starts without retraining
- Model and tokenizer paths resolve from `Path(__file__)`
- Sample inference completes within acceptable memory and time
- No proprietary code appears in examples or logs
- Legacy checkpoint is not presented as an attention checkpoint
