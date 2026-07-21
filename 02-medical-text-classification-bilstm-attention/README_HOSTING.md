# Hosting Guide

## Recommended option: Streamlit Community Cloud

Streamlit Community Cloud is the best first hosting option for this portfolio project because the application is already written in Streamlit, the model artifact is small, the source is stored on GitHub, and no external database or secret is required.

## Required repository files

```text
bi-directional-lstm-projects/
├── .streamlit/
│   └── config.toml
└── 02-medical-text-classification-bilstm-attention/
    ├── app/
    │   └── streamlit_app.py
    ├── models/
    │   ├── medical_text_bilstm_attention_model.keras
    │   ├── tokenizer_config.json
    │   ├── label_mapping.json
    │   └── model_metadata.json
    ├── src/
    └── requirements.txt
```

The app loads pre-trained artifacts. Training is not run during startup.

## Deployment steps

1. Push the complete monorepo to GitHub.
2. Confirm the app works locally:

   ```bash
   cd 02-medical-text-classification-bilstm-attention
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app/streamlit_app.py
   ```

3. Sign in to Streamlit Community Cloud using the GitHub account that can access the repository.
4. Create a new app.
5. Select:
   - repository: `YOUR_USERNAME/bi-directional-lstm-projects`
   - branch: `main`
   - entrypoint: `02-medical-text-classification-bilstm-attention/app/streamlit_app.py`
6. Open advanced settings and select **Python 3.11**.
7. Choose a clear subdomain such as:
   - `medical-text-bilstm-attention`
   - `anmol-medical-nlp-bilstm`
8. Deploy.
9. Test all five safe examples and a small non-sensitive CSV.
10. Add the final `streamlit.app` link to the project README, root README, resume, LinkedIn, and portfolio.

## Dependency notes

Community Cloud must be able to find a dependency file. This project keeps `requirements.txt` in the project folder, above the nested app entrypoint, and all imported project modules are referenced relative to the project root.

The runtime dependencies are pinned or bounded for reproducibility. The project targets Python 3.11.

## Updating the app

Push updates to the connected GitHub branch. Streamlit Community Cloud rebuilds when dependency files change and refreshes the app when source files change.

## Common troubleshooting

### Model does not load

- Confirm all four required model files exist.
- Confirm the `.keras` file was not corrupted during upload.
- Confirm TensorFlow installed successfully.
- Run:

  ```bash
  python scripts/validate_artifacts.py
  ```

### Module not found

- Confirm `src/` exists inside the same project folder.
- Confirm deployment entrypoint is exactly:
  `02-medical-text-classification-bilstm-attention/app/streamlit_app.py`
- Confirm external packages are declared in `requirements.txt`.

### Build is slow

TensorFlow is a large dependency. Initial deployment can take longer than a lightweight Streamlit dashboard. Subsequent source-only updates are normally faster.

## Docker option

```bash
docker build -t medical-text-bilstm .
docker run --rm -p 8501:8501 medical-text-bilstm
```

Open `http://localhost:8501`.

## Responsible public hosting

- Keep the medical disclaimer visible.
- Do not log or persist uploaded text.
- Do not request diagnoses or clinical decisions.
- Do not accept real patient identifiers.
- Clearly label the supplied model as a ten-row demonstration artifact.
- Replace the model only after licensing, privacy, bias, and validation reviews.

## Official references

- Streamlit Community Cloud deployment:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- App dependencies:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- File organization:
  https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
