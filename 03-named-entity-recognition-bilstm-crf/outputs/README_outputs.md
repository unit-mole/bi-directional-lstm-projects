# Output artifacts

Files prefixed with `legacy_` were reconstructed from the recorded outputs in the supplied notebook and correspond to the supplied **BiLSTM-softmax** model, not a CRF-trained model.

After running `python scripts/train_model.py` and `python scripts/evaluate_model.py`, the project generates unprefixed production artifacts such as:

- `training_history.csv`
- `training_curve.png`
- `entity_level_classification_report.csv`
- `token_level_classification_report.csv`
- `confusion_matrix.png`
- `error_analysis.csv`
- `model_metrics.json`

The repository does not fabricate a CRF result table before CRF training has actually been run.
