# Project Audit

## Supplied Artifact Findings

- Dataset: 10 rows, 2 columns (`text`, `emotion`).
- Raw labels: joy, anger, fear, sadness, surprise, calm.
- `MIN_CLASS_COUNT=2` removed anger, surprise, and calm.
- Final retained dataset: 7 rows and 3 labels.
- Split sizes: 4 training, 1 validation, 2 test.
- Tokenizer document count: 7; vocabulary size used by model: 40.
- Model: Embedding → Bidirectional LSTM (`return_sequences=False`) → Dense.
- Attention layer: absent.
- Test accuracy: 0.50 on two rows.
- Weighted F1: 0.33 on two rows.
- Streamlit code: placeholder only.

## Release Assessment

The supplied checkpoint is not suitable for portfolio performance claims. It is suitable only as a reproducibility artifact showing the starting point. The repository code is release-ready as a project framework, while final model readiness depends on retraining with a complete licensed dataset and replacing the legacy artifact set.
