# Medical Text Classification using BiLSTM with Attention

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](#local-setup)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](#technology-stack)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red.svg)](#streamlit-demo)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-success.svg)](../../actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

An end-to-end healthcare NLP portfolio project that classifies clinical-style text into medical specialty categories using a **Bidirectional LSTM with temporal attention**. The project includes reusable preprocessing and training modules, saved model artifacts, class probabilities, optional attention terms, error-analysis outputs, a TF-IDF baseline, a professional Streamlit application, Docker support, automated tests, and GitHub Actions CI.

> [!CAUTION]
> **Medical and privacy disclaimer:** This project is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool and must not be used to diagnose, treat, prevent, or manage any medical condition. Medical-text classification requires clinical validation, domain expertise, and qualified professional review. Do not upload private, sensitive, confidential, or personally identifiable health information. Predictions are machine-learning outputs, not medical advice.

## Live demo

**Streamlit:** `Add deployed Streamlit URL here`

## Portfolio one-liner

> Built a privacy-aware healthcare NLP pipeline using a Bidirectional LSTM with temporal attention to classify clinical-style text into five medical specialties, with probability outputs, attention terms, batch inference, Streamlit deployment, Docker, tests, and CI.

## Problem statement

Healthcare and quality systems generate large volumes of unstructured text: notes, complaint narratives, issue descriptions, failure comments, service records, and customer feedback. Manual routing and categorization can be slow and inconsistent.

This project asks:

> **Given a medical text passage, can a sequence model classify it into the correct medical specialty?**

The model assigns one of five labels:

- Cardiology
- Gastroenterology
- Neurology
- Orthopedic
- Radiology

## Why a Bidirectional LSTM with attention?

A standard LSTM reads a sequence in one direction. A Bidirectional LSTM combines:

- a forward representation that processes text from beginning to end, and
- a backward representation that processes it from end to beginning.

This helps the model use context on both sides of a token. The attention layer then learns a relative weight for each sequence step and forms a weighted context vector instead of depending only on one final hidden state.

```text
Medical text
    ↓
Text cleaning
    ↓
Tokenizer + integer sequences
    ↓
Post-padding / post-truncation
    ↓
Embedding layer
    ↓
Bidirectional LSTM
    ↓
Temporal attention
    ↓
Dense + dropout
    ↓
Five-class softmax probabilities
```

Attention weights can support inspection of which tokens received larger internal weights, but they are **not** validated clinical explanations and must not be interpreted as causality.

## Actual supplied dataset

The attached file was inspected before this project was generated.

| Attribute | Actual value |
|---|---|
| File | `sample_medical_text_data.csv` |
| Original text column | `transcription` |
| Original target column | `medical_specialty` |
| Rows | 10 |
| Classes | 5 |
| Rows per class | 2 |
| Missing text / label | 0 / 0 |
| Duplicate text | 0 |
| Average text length | 11.1 words |
| Maximum text length | 15 words |

The included rows are short synthetic examples. They are safe for demonstrating the pipeline, but far too small for credible model training or benchmarking.

See [`data/README_data.md`](./data/README_data.md) for replacement-dataset and privacy guidance.

## Honest audit of the supplied model

The original notebook created a functional Keras model and saved the model, tokenizer, label mapping, training history, and prediction analysis. The underlying architecture contains 128,049 trainable parameters.

### Original split

| Split | Rows |
|---|---:|
| Training | 5 |
| Validation | 2 |
| Test | 3 |

### Reported holdout results

| Metric | Value |
|---|---:|
| Test accuracy | 0.3333 |
| Macro F1 | 0.1000 |
| Weighted F1 | 0.1667 |
| Weighted precision | 0.1111 |
| Weighted recall | 0.3333 |
| Test loss | 1.5904 |

The saved model predicted **Orthopedic for all three test rows**, with near-uniform probabilities of roughly 20% per class. Therefore:

- the artifact verifies that the model pipeline runs,
- it does **not** establish useful generalization,
- the metrics must not be promoted as a model benchmark,
- a larger and representative dataset is the highest-priority improvement.

Read [`PROJECT_AUDIT.md`](./PROJECT_AUDIT.md) for the full assessment.

## Medical-text preprocessing

The project avoids a one-size-fits-all cleaner.

### `legacy` mode

This exactly reproduces the supplied artifact’s preprocessing:

- lowercase,
- normalize whitespace,
- remove punctuation,
- retain letters and numbers.

The Streamlit app uses this mode because inference preprocessing must match training.

### `clinical_safe` mode

This is recommended when retraining:

- Unicode normalization,
- HTML removal,
- whitespace normalization,
- lowercase by default,
- preserve numbers and clinically meaningful notation,
- preserve decimals, percentages, slashes, colons, plus/minus, and hyphens,
- preserve negations such as `no`, `not`, and `denies`,
- optional conservative abbreviation mapping.

Examples such as `BP 140/90`, `O2 98%`, and `2.5-mg` retain more context.

## Class imbalance

The ten-row sample is exactly balanced, so its computed class weights are all 1.0. The reusable training pipeline still computes balanced class weights for replacement datasets.

For medical or quality text, accuracy alone can hide poor minority-class performance. The project therefore supports:

- macro precision, recall, and F1,
- weighted precision, recall, and F1,
- per-class precision, recall, and F1,
- confusion matrix,
- top-k accuracy,
- error analysis,
- class-weighted training.

Macro F1 gives each class equal influence. Per-class recall is important when a small but operationally important category is easy to miss.

## Baseline comparison

A TF-IDF + Logistic Regression baseline is implemented in [`src/baseline.py`](./src/baseline.py).

The included sample produces:

| Model | Evaluation protocol | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 2-fold stratified CV | 0.1000 | 0.1000 | 0.1000 |
| Supplied BiLSTM + Attention | 3-row holdout | 0.3333 | 0.1000 | 0.1667 |

These rows use different evaluation protocols and are not directly comparable. Both results are unstable because the dataset has only ten rows. The table exists to demonstrate the baseline workflow, not to claim deep-learning superiority.

## Streamlit demo

The app supports:

- manual medical-text entry,
- five safe synthetic examples,
- predicted category,
- confidence score,
- top-three category probabilities,
- optional attention-term weights,
- CSV batch upload,
- selectable text column,
- batch result table,
- predicted-category distribution,
- downloadable scored CSV,
- model architecture and limitations,
- visible medical and privacy disclaimers.

The app loads the pre-trained model and never retrains during startup.

### Example output contract

```text
Input medical text
Predicted category
Confidence
Top 3 category probabilities
Important terms from attention, when available
Interpretation
Medical disclaimer
```

## Project structure

```text
02-medical-text-classification-bilstm-attention/
├── .streamlit/
│   └── README.md
├── app/
│   ├── __init__.py
│   └── streamlit_app.py
├── archive/
│   └── Medical_Text_Classification_using_Bi_Directional_LSTM_COMPLETE_corrected.ipynb
├── data/
│   ├── README_data.md
│   └── sample_medical_text_data.csv
├── images/
│   └── README.md
├── models/
│   ├── label_encoder.pkl
│   ├── label_mapping.json
│   ├── medical_text_bilstm_attention_model.keras
│   ├── model_metadata.json
│   └── tokenizer_config.json
├── notebooks/
│   └── medical_text_classification_bilstm_attention.ipynb
├── outputs/
│   ├── baseline_comparison.csv
│   ├── baseline_metrics.csv
│   ├── baseline_prediction_analysis.csv
│   ├── class_distribution.png
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── data_quality_summary.json
│   ├── medical_text_prediction_analysis.csv
│   ├── medical_text_training_history.csv
│   ├── model_architecture.png
│   ├── model_metrics.json
│   └── training_curve.png
├── scripts/
│   ├── evaluate_model.py
│   ├── run_streamlit.py
│   ├── train_baseline.py
│   ├── train_model.py
│   └── validate_artifacts.py
├── src/
│   ├── artifacts.py
│   ├── attention_layer.py
│   ├── baseline.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── inference_pipeline.py
│   ├── medical_text_prediction.py
│   ├── medical_text_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── sequence_generation.py
│   ├── tokenizer_utils.py
│   └── visualization.py
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── FILE_MANIFEST.csv
├── IMPROVEMENTS.md
├── LICENSE
├── MONOREPO_INTEGRATION.md
├── PROJECT_AUDIT.md
├── README.md
├── README_HOSTING.md
├── requirements-dev.txt
├── requirements.txt
├── run_local.bat
├── run_local.sh
└── train_model.py
```

## Local setup

### Windows PowerShell

```powershell
cd bi-directional-lstm-projects\02-medical-text-classification-bilstm-attention
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app\streamlit_app.py
```

Or double-click / run:

```powershell
run_local.bat
```

### macOS or Linux

```bash
cd bi-directional-lstm-projects/02-medical-text-classification-bilstm-attention
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Or:

```bash
chmod +x run_local.sh
./run_local.sh
```

## Artifact validation

Metadata-only validation does not import TensorFlow:

```bash
python scripts/validate_artifacts.py --metadata-only
```

Full validation loads the tokenizer and Keras model:

```bash
python scripts/validate_artifacts.py
```

## Train a replacement model

```bash
python scripts/train_model.py \
  --data /path/to/deidentified_medical_text.csv \
  --text-column transcription \
  --label-column medical_specialty \
  --epochs 20 \
  --batch-size 32 \
  --preprocessing-mode clinical_safe
```

Artifacts are saved under `models/`; evaluation and visual outputs are saved under `outputs/`.

Do not use the ten-row sample to claim a trained production-quality model.

## Run the baseline

```bash
python scripts/train_baseline.py
```

## Evaluate a saved model

```bash
python scripts/evaluate_model.py \
  --data /path/to/labeled_evaluation_data.csv \
  --text-column transcription \
  --label-column medical_specialty
```

Use an evaluation dataset that is independent of training data.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Skip the model-loading test:

```bash
pytest -q -m "not model"
```

## Docker

```bash
docker build -t medical-text-bilstm-attention .
docker run --rm -p 8501:8501 medical-text-bilstm-attention
```

Open `http://localhost:8501`.

## Deployment

The recommended first deployment target is Streamlit Community Cloud.

Entrypoint:

```text
02-medical-text-classification-bilstm-attention/app/streamlit_app.py
```

Use Python 3.11 and keep `requirements.txt` in the project folder. Detailed instructions are available in [`README_HOSTING.md`](./README_HOSTING.md).

## Limitations

- The supplied data contains only ten synthetic rows.
- The five-row training split cannot support reliable learning.
- The test set contains only three rows.
- Attention weights are not clinically validated explanations.
- The label space contains only five specialties.
- Medical language varies across institutions, specialties, populations, and documentation practices.
- The included model has not been tested for bias, calibration, robustness, dataset shift, or external validity.
- The app is not designed to store or process protected health information.
- Clinical deployment would require governance, security, privacy, quality controls, monitoring, and qualified human review.

## Future improvements

1. Replace the sample with an appropriately licensed, de-identified dataset containing meaningful support per class.
2. Use patient/document-level grouped splits to prevent leakage.
3. Add repeated stratified validation or a fixed external holdout.
4. Compare against TF-IDF, linear SVM, BiLSTM without attention, and transformer baselines.
5. Add probability calibration and confidence-based abstention.
6. Evaluate class-specific recall and confusion costs with domain experts.
7. Explore domain embeddings or clinical-language models while retaining the BiLSTM portfolio objective.
8. Add model-card, data-card, bias review, and drift-monitoring templates.
9. Add automated de-identification checks before batch scoring.
10. Replace the demonstration artifact and update screenshots only after credible validation.

## Skills demonstrated

- Healthcare NLP
- Medical-text preprocessing
- Multi-class classification
- Bidirectional LSTM modeling
- Custom attention mechanism
- TensorFlow/Keras artifact serialization
- Class imbalance handling
- Macro and weighted F1 evaluation
- Error analysis and baseline comparison
- Streamlit app development
- Batch inference and downloadable outputs
- Docker, pytest, and GitHub Actions
- Responsible AI and privacy-aware communication

## Connection to Quality Data Science

The technical pattern extends naturally beyond healthcare text:

- complaint and GCS case categorization,
- issue-description routing,
- failure-mode text classification,
- customer feedback analysis,
- root-cause narrative triage,
- service-note categorization,
- quality-event prioritization,
- automated insight generation.

This makes the project relevant to both the current Quality Data Scientist role and future Data Science, NLP, ML, and Applied AI positions.

## GitHub positioning

### Repository description

> End-to-end BiLSTM portfolio featuring emotion detection, medical-text classification, NER with CRF, Siamese semantic matching, attention mechanisms, TensorFlow/Keras models, and interactive Streamlit demos.

### Suggested repository topics

```text
bidirectional-lstm
bilstm
deep-learning
nlp
healthcare-nlp
medical-text-classification
sequence-modeling
attention-mechanism
tensorflow
keras
streamlit
text-classification
named-entity-recognition
siamese-network
machine-learning
data-science
portfolio-projects
```

## License

Code is released under the MIT License. Dataset and model usage remain subject to source licensing, privacy obligations, and any applicable organizational policies.
