# Improvements Made

- Converted the notebook into modular source files.
- Preserved the original notebook under `archive/`.
- Normalized the double-encoded tokenizer JSON.
- Added model metadata and artifact validation.
- Added a real Streamlit app with manual, batch, and ranking modes.
- Added conservative text preprocessing and column detection.
- Added unordered-pair deduplication and training-only tokenizer fitting.
- Added class weights, validation-based threshold tuning, and richer metrics.
- Added a TF-IDF baseline implementation for future comparison.
- Added tests, Docker, GitHub Actions, local launchers, and hosting documentation.
- Added charts generated from the actual supplied data and outputs.
- Added prominent limitations and responsible-use language.
- Prevented training on fewer than 100 rows to avoid misleading portfolio claims.
