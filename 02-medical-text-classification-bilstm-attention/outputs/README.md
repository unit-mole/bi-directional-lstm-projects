# Output artifacts

These files were generated from the supplied ten-row demonstration dataset and model artifact.

## Tracked diagnostic outputs

- `class_distribution.png`
- `text_length_distribution.png`
- `model_architecture.png`
- `training_curve.png`
- `confusion_matrix.png`
- `classification_report.csv`
- `per_class_performance.csv`
- `model_metrics.json`
- `sample_predictions.csv`
- `error_analysis.csv`
- `baseline_metrics.csv`
- `baseline_prediction_analysis.csv`
- `baseline_comparison.csv`
- `data_quality_summary.json`

## Important limitation

The reported model holdout contains only three rows. These outputs confirm the pipeline and expose its weaknesses; they are not reliable benchmark evidence.

## Attention visualization

A static attention plot is intentionally not fabricated. The Streamlit app attempts to extract attention terms from the saved model at runtime. When a replacement model is trained with TensorFlow, add a reproducible attention visualization generated from non-sensitive sample text.
