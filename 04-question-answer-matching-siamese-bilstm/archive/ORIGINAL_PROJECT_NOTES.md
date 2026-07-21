# Original Project Archive

The original notebook and saved CSV outputs are preserved here without modification.

Key issues identified:

1. Only 15 synthetic rows were used.
2. The tokenizer was fitted before splitting, creating vocabulary leakage.
3. The test set contained only 3 rows.
4. The model reported accuracy only during training.
5. Predictions were almost constant near 0.51.
6. The optional Streamlit cell was a non-functional placeholder.
7. No reusable package structure, tests, CI, Docker, metadata, threshold tuning, or deployment guide existed.
8. The task was duplicate-question detection, not factual question-answer validation.

The GitHub-ready project retains the original artifact but separates it from improved modular code.
