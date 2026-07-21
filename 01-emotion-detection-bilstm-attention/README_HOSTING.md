# Hosting Guide — Streamlit Community Cloud

## Why Streamlit Community Cloud

The application is already written in Streamlit, the repository can be connected directly from GitHub, and the deployed app receives a shareable `streamlit.app` address. The deployment should load saved artifacts and must never retrain during startup.

## Required Files

Commit these files:

- `app/streamlit_app.py`
- `requirements.txt`
- `src/`
- `.streamlit/config.toml`
- a complete model artifact set in `models/`

For the final project, the preferred set is:

- `models/emotion_bilstm_attention_model.keras`
- `models/tokenizer.json`
- `models/label_mapping.json`
- `models/model_metadata.json`

The supplied legacy artifacts also let the app start, but they should remain visibly labeled as a limited demonstration.

## Deployment Steps

1. Push `bi-directional-lstm-projects` to GitHub.
2. Confirm the model file is within GitHub/host limits. Use Git LFS when needed.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Select **Create app**.
5. Choose the repository and branch.
6. Set the main file path to:

```text
01-emotion-detection-bilstm-attention/app/streamlit_app.py
```

7. Open advanced settings and select Python 3.11 when available.
8. Deploy and review the build logs.
9. Test manual prediction, CSV upload, download, disclaimers, and mobile layout.
10. Replace README placeholders with the final `https://<app-name>.streamlit.app` URL.

## Monorepo Dependency Note

The `requirements.txt` file is stored inside the project folder. If the platform does not automatically discover it from the app's parent project directory, use one of these approaches:

- configure the project folder as the app root when the deployment UI supports it, or
- place a deployment-only `requirements.txt` at the repository root containing:

```text
-r 01-emotion-detection-bilstm-attention/requirements.txt
```

## Pre-deployment Checklist

- [ ] Replace `USERNAME` and demo URL placeholders.
- [ ] Train and include the true attention-model artifacts.
- [ ] Verify dataset licensing.
- [ ] Remove private or confidential data.
- [ ] Confirm the app does not train at startup.
- [ ] Run `pytest -q` locally.
- [ ] Run `python scripts/validate_project.py`.
- [ ] Test batch CSV scoring and download.
- [ ] Confirm responsible-use language is visible.

## Hugging Face Spaces Alternative

Create a Streamlit Space, upload this project folder, set `app/streamlit_app.py` as the entry point, and retain the same artifact-loading behavior. Streamlit Community Cloud remains the simplest first deployment for this repository.
