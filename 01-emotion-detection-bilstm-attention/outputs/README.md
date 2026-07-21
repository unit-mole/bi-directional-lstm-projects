# Output Artifacts

Files prefixed with `legacy_` were reconstructed from the supplied notebook outputs and are retained for auditability. They should **not** be presented as final performance because the test set contained only two rows.

After training the upgraded attention architecture on a complete dataset, this folder will receive:

- `class_distribution.png`
- `text_length_distribution.png`
- `training_accuracy_curve.png`
- `training_loss_curve.png`
- `confusion_matrix.png` and `.csv`
- `classification_report.csv`
- `prediction_analysis.csv`
- `misclassified_examples.csv`
- `model_metrics.json`
- baseline comparison outputs

Attention-token values are shown interactively in Streamlit after an actual attention model is trained.
