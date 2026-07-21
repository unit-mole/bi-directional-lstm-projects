# Improvements made

- Reclassified the supplied model honestly as a BiLSTM-softmax baseline.
- Added a pure TensorFlow linear-chain CRF implementation with trainable transitions.
- Added sequence log-likelihood, masked training, and Viterbi decoding.
- Removed dependency on end-of-life TensorFlow Addons.
- Added CoNLL, CSV, and Hugging Face data loaders.
- Added BIO validation, deterministic repair, safe padding, OOV handling, and mapping persistence.
- Added human-readable span extraction, token offsets, highlighting, confidence display, and batch output.
- Added entity-level seqeval metrics, token reports, confusion matrix generation, and error analysis.
- Added a polished Streamlit interface with single-text, batch CSV, and CoNLL validation workflows.
- Added responsible-use and privacy warnings.
- Added Docker, Windows/Linux launchers, tests, GitHub Actions, hosting instructions, and file manifest.
- Preserved the original notebook under `archive/` and created a clean modular notebook.
