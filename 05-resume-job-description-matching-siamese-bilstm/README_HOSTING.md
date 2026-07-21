# Hosting Guide

## Recommended option: Streamlit Community Cloud

The project is already structured around `app/streamlit_app.py`, so Streamlit Community Cloud is the simplest portfolio deployment target.

### Required files

- `app/streamlit_app.py`
- `requirements.txt`
- `models/resume_job_siamese_bilstm_model.keras`
- `models/tokenizer.json`
- `models/model_metadata.json`
- `data/sample/sample_resume_job_pairs.csv`
- `src/` package
- `.streamlit/config.toml`

### Deployment steps

1. Push the complete monorepo to GitHub.
2. Confirm the model file is below GitHub's regular file-size limit. Use Git LFS if a future model becomes larger.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app and select the `bi-directional-lstm-projects` repository.
5. Set the main file path to:
   `05-resume-job-description-matching-siamese-bilstm/app/streamlit_app.py`
6. Select Python 3.11 when the platform offers a runtime choice.
7. Deploy and review the build logs.
8. Test all four tabs with synthetic data only.
9. Add the final app URL to the project README, root README, LinkedIn, resume, and portfolio.

### Operational notes

- The app loads pre-trained artifacts and never retrains at startup.
- TensorFlow can make the first deployment slower. Keep the model small and avoid unnecessary packages.
- Do not store private credentials in code. Use Streamlit secrets only when a future integration requires them.
- Add a clear privacy notice and disable logging of raw resume text in any real deployment.

## Docker deployment

```bash
docker build -t resume-jd-bilstm .
docker run --rm -p 8501:8501 resume-jd-bilstm
```

Open `http://localhost:8501` in a browser.

## Hugging Face Spaces alternative

Create a Streamlit Space, upload the project files, ensure the app entry point is configured, and retain the same model, tokenizer, metadata, and responsible-use notice. Streamlit Community Cloud remains the more direct choice for this repository layout.
