# Improvements Made

## Structure and engineering

- Converted the notebook-only workflow into reusable modules under `src/`.
- Added dedicated scripts for training, evaluation, baseline modeling, artifact validation, and Streamlit execution.
- Added project-level tests, Docker support, GitHub Actions CI, local run scripts, and a file manifest.
- Preserved the original notebook under `archive/`.
- Added a clean modular notebook under `notebooks/`.

## Data handling

- Added automatic text and label column detection using the actual columns supplied.
- Added missing-value, empty-text, duplicate-text, and rare-class checks.
- Added a structured data audit and saved data-quality summary.
- Added clear privacy and dataset-redistribution guidance.
- Kept the included ten-row file as a safe synthetic sample.

## Medical text preprocessing

- Added Unicode normalization and HTML cleanup.
- Added a `legacy` mode that exactly matches the supplied model.
- Added a `clinical_safe` mode for retraining that preserves values such as `140/90`, `98%`, and `2.5-mg`.
- Preserved negation terms rather than treating them as stopwords.
- Made abbreviation expansion explicit and opt-in because abbreviations can be ambiguous.

## Model and attention

- Moved the custom attention layer into a serializable Keras layer module.
- Added class-weight support, top-k accuracy, callbacks, best-model checkpoints, and reusable training configuration.
- Added a metadata artifact describing labels, sequence length, vocabulary, preprocessing mode, architecture, split sizes, and limitations.
- Added optional extraction of token attention weights for the Streamlit interface.
- Added robust loading of the tokenizer’s original nested JSON-string format.

## Evaluation

- Added accuracy, macro/weighted precision, recall, and F1.
- Added classification report, confusion matrix, error-analysis, and baseline outputs.
- Added a TF-IDF + Logistic Regression baseline script.
- Clearly marked that metrics from the ten-row demonstration are not statistically meaningful.
- Prevented the project from overstating clinical or predictive performance.

## Streamlit application

- Added functional single-text inference.
- Added safe synthetic examples.
- Added top-three class probabilities.
- Added optional attention-term display.
- Added CSV batch upload, text-column selection, scoring, distribution chart, and CSV download.
- Added medical, privacy, and responsible-use disclaimers.
- Added model details and an honest limitation section.
- Ensured the app loads saved artifacts and never retrains at startup.

## Deployment and quality controls

- Added current Streamlit Community Cloud instructions.
- Added a Python 3.11 Docker image.
- Added a lightweight CI workflow that compiles code, validates artifacts, runs non-model tests, and checks app import.
- Added explicit `.gitignore` rules for private/full datasets, secrets, checkpoints, and runtime uploads.
