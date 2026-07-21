# Hosting guide

## Recommended option: Streamlit Community Cloud

This project is a natural fit for Streamlit Community Cloud because the interface is already implemented in Streamlit and the model artifact is below GitHub's normal 100 MB single-file limit.

### Before deployment

1. Push the repository to GitHub.
2. Confirm these files are committed:
   - `03-named-entity-recognition-bilstm-crf/app/streamlit_app.py`
   - `03-named-entity-recognition-bilstm-crf/requirements.txt`
   - `03-named-entity-recognition-bilstm-crf/models/legacy_bilstm_softmax_model.h5`
   - all mapping files and `model_metadata.json`
3. Run locally from the monorepo root:

```bash
streamlit run 03-named-entity-recognition-bilstm-crf/app/streamlit_app.py
```

### Deploy

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app**.
3. Select the `bi-directional-lstm-projects` repository and branch.
4. Set the entrypoint to:

```text
03-named-entity-recognition-bilstm-crf/app/streamlit_app.py
```

5. In Advanced settings, select **Python 3.12**.
6. Deploy and test all tabs.
7. Add the generated `*.streamlit.app` URL to the project README, root README, resume, LinkedIn, and portfolio.

The dependency file is beside the project and Streamlit can resolve it for a subdirectory entrypoint. The app loads pretrained artifacts and never trains during startup.

## Resource note

TensorFlow can be memory-intensive on free hosting. Keep one model cached with `st.cache_resource`, limit batch uploads, and avoid loading the training dataset in the app. The included app follows these practices.

## Hugging Face Spaces alternative

Create a Streamlit Space, copy the project files into the Space repository, set `app/streamlit_app.py` as the app entrypoint as required by the current Spaces configuration, and retain the same model artifacts. Streamlit Community Cloud is simpler for this monorepo because it deploys directly from the existing GitHub repository.
