# Streamlit Community Cloud Deployment

## App settings

- Repository: `unit-mole/bi-directional-lstm-projects`
- Branch: `main`
- Main file: `01-emotion-detection-bilstm-attention/app/streamlit_app.py`
- Python version: `3.11`
- Secrets: leave blank

The file `app/requirements.txt` is deliberately stored beside the Streamlit entrypoint so Community Cloud installs the Project 01 dependencies instead of searching the monorepo root.

## Redeploy after pushing

1. Push the replacement folder and workflow to `main`.
2. Open the existing app.
3. Select **Manage app**.
4. Select **Reboot app** if the app does not refresh automatically.
5. Confirm the Model Snapshot sidebar shows six classes and the bundled model metrics.

## Smoke checks

- Joy sample returns Joy.
- Fear sample returns Fear.
- An attention chart appears after prediction.
- Batch CSV scoring returns a downloadable CSV.
