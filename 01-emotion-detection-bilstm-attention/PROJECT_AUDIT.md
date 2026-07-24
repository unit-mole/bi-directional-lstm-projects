# Project Audit

## Resolved issues

- Removed the deployed three-class legacy checkpoint with near-uniform probabilities.
- Replaced it with a trained six-class BiLSTM + temporal-attention checkpoint.
- Added `app/requirements.txt` for Streamlit monorepo deployment.
- Added token-level attention output.
- Added balanced class support, stratified splits, class weighting, early stopping, gradient clipping, and artifact validation.
- Added project-specific GitHub Actions path filters.

## Validation snapshot

- Dataset rows: 7,200
- Train rows: 5,040
- Validation rows: 1,080
- Test rows: 1,080
- Classes: anger, fear, joy, love, sadness, surprise
- Packaged test accuracy: 100.0%
- Packaged macro F1: 100.0%

## Important limitation

The packaged metrics come from a template-augmented educational dataset and are expected to be optimistic. They verify that the end-to-end model, attention mechanism, artifacts, and application work correctly. They are not a substitute for evaluation on naturally occurring text.
