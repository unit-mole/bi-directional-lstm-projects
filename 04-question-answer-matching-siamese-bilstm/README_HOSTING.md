# Hosting Guide

## Recommended Option: Streamlit Community Cloud

Streamlit Community Cloud is the simplest fit because the interface is already written in Streamlit and the app only needs to load committed model artifacts.

## Required Files

- `app/streamlit_app.py`
- `requirements.txt`
- `models/qa_siamese_bilstm_model.keras`
- `models/tokenizer.json`
- `models/model_metadata.json`
- `src/`
- `.streamlit/config.toml`

## Deployment Steps

1. Push this project into the public `bi-directional-lstm-projects` GitHub repository.
2. Confirm the model file is present and below GitHub's normal file-size limit.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Select the repository and branch.
5. Set the app file to:

   ```text
   04-question-answer-matching-siamese-bilstm/app/streamlit_app.py
   ```

6. Deploy and review the build log.
7. Test manual prediction, batch CSV scoring, and ranking.
8. Add the generated public URL to the project README, root README, resume, LinkedIn, and portfolio.

## Local Deployment Check

```bash
pip install -r requirements.txt
python scripts/validate_artifacts.py
streamlit run app/streamlit_app.py
```

## Hugging Face Spaces Alternative

Create a Streamlit Space, upload the project contents, keep the entry point at `app/streamlit_app.py`, and use the same `requirements.txt`. Streamlit Community Cloud remains easier for a GitHub-centered portfolio.

## Troubleshooting

- Model-load error: confirm TensorFlow/Keras compatibility and all three model artifacts.
- Memory error: reduce dependencies or use a smaller retrained model.
- File-path error: keep the repository folder structure unchanged.
- Slow cold start: TensorFlow initialization is the main contributor; cached loading prevents repeated reloads per session.
