# Emotion Detection using Bidirectional LSTM with Attention

[![CI](https://github.com/USERNAME/bi-directional-lstm-projects/actions/workflows/01-emotion-detection-bilstm-attention.yml/badge.svg)](https://github.com/USERNAME/bi-directional-lstm-projects/actions/workflows/01-emotion-detection-bilstm-attention.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](#local-setup)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](#technology-stack)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red)](#streamlit-demo)

> An end-to-end, recruiter-friendly NLP project that classifies text into emotion categories using a **Bidirectional LSTM with temporal attention**, produces class probabilities and error-analysis outputs, and supports interactive single-text and batch-CSV inference through Streamlit.

## Responsible Use and Privacy

**This project is for educational and portfolio demonstration purposes only.** Emotion detection models may misinterpret tone, sarcasm, cultural context, mixed emotions, and ambiguous language. Do not upload private, sensitive, confidential, or personally identifiable text. Predictions are model estimates—not definitive emotional truth—and must not be used as the sole basis for mental-health, hiring, insurance, legal, surveillance, or other high-stakes decisions.

## Problem Statement

Given a sentence, message, review, support ticket, or feedback comment, can a sequence model identify the most likely emotion expressed in the text?

The system returns:

- predicted emotion,
- confidence score,
- complete class-probability distribution,
- top competing emotions,
- token-level attention weights when the upgraded model is trained,
- a plain-language interpretation and responsible-use reminder.

## Why This Project Matters

Fine-grained emotion classification extends beyond positive/negative sentiment. It can support complaint-text analytics, customer feedback monitoring, case-comment triage, support-ticket prioritization, conversational AI, and automated quality insight generation. These use cases connect directly with a Quality Data Scientist's experience transforming unstructured operational text into measurable signals.

## Honest Audit of the Supplied Project

The original notebook and artifacts were reviewed before this repository was generated.

| Finding | Supplied implementation | Portfolio-ready correction |
|---|---|---|
| Dataset | 10-row placeholder CSV | Dynamic loader plus full-dataset instructions and safe sample |
| Labels | Six labels in raw sample; singleton labels removed | Preserve labels dynamically and require sufficient class support |
| Retained data | 7 rows after `MIN_CLASS_COUNT=2` | Fail clearly rather than silently presenting weak results |
| Split | 4 train / 1 validation / 2 test | One aligned stratified dataframe split on adequate data |
| Tokenizer | Fitted before train/test split | Fitted on training text only to reduce leakage |
| Architecture | Embedding → BiLSTM → Dense | Embedding → BiLSTM sequences → **temporal attention** → Dense |
| Attention | Not present | Serializable custom attention layer with optional token weights |
| Class imbalance | Not handled | Balanced class weights plus macro/weighted F1 reporting |
| Streamlit | Placeholder code only | Full manual input, samples, CSV batch scoring, charts, downloads |
| Testing/CI | Not present | Unit tests, compile checks, import validation, GitHub Actions |
| Performance claims | 50% accuracy on 2 test rows | Clearly marked as statistically unreliable legacy output |

The supplied label mapping contains only `fear`, `joy`, and `sadness`, and the bundled tokenizer was fitted on seven documents. The app can load this checkpoint in **legacy demonstration mode**, but it does not label that artifact as an attention model.

## Intended Model Architecture

```text
Raw text
   ↓
Emotion-aware text normalization
   ↓
Training-only Keras tokenizer + OOV token
   ↓
Post-padded integer sequence
   ↓
Trainable embedding layer
   ↓
Bidirectional LSTM (return_sequences=True)
   ↓
Temporal attention across sequence positions
   ↓
Dense layer + dropout
   ↓
Softmax probability distribution across emotion classes
```

### Why Bidirectional LSTM?

The forward LSTM captures left-to-right context while the backward LSTM captures right-to-left context. This is useful for phrases whose emotional meaning depends on words appearing before and after an important token.

### Why Attention?

The attention layer learns a normalized weight for every sequence position and creates a weighted context vector. This lets the classifier focus on emotionally informative tokens such as *excited*, *afraid*, *frustrated*, or *grateful* rather than relying only on a final hidden state.

## Dataset

The bundled `data/emotion_dataset.csv` is the supplied **10-row placeholder**, with columns `text` and `emotion`. It contains six labels: joy, anger, fear, sadness, surprise, and calm. It is intentionally retained for reproducibility and UI checks, not for credible training.

Use a complete licensed emotion dataset for the final model. The data loader supports common text columns (`text`, `sentence`, `message`, `tweet`, `content`, `comment`) and target columns (`emotion`, `label`, `target`, `class`, `sentiment`). See [`data/README_data.md`](./data/README_data.md).

## Text Preprocessing

The upgraded pipeline:

- normalizes Unicode and HTML entities,
- removes HTML tags,
- replaces URLs and user mentions with stable tokens,
- retains hashtag words,
- converts emojis to textual aliases when possible,
- represents repeated `!` and `?` as intensity tokens,
- records all-uppercase emphasis,
- avoids aggressive stop-word removal or stemming,
- fits the tokenizer on the training partition only,
- uses an explicit `<OOV>` token,
- applies identical preprocessing during training and inference.

## Class Imbalance

The training script computes balanced class weights from the training split. Evaluation reports both macro F1 and weighted F1:

- **Macro F1** gives every emotion class equal importance and reveals poor minority-class performance.
- **Weighted F1** incorporates class support and summarizes overall performance under imbalance.
- **Per-class recall** helps identify emotions the model regularly misses.

## Evaluation

Generated outputs include accuracy, macro precision/recall/F1, weighted F1, per-class classification report, confusion matrix, training curves, prediction analysis, and high-confidence errors.

### Supplied Legacy Results — Do Not Treat as Final

| Metric | Value | Reliability warning |
|---|---:|---|
| Test accuracy | 0.50 | Only 2 test rows |
| Weighted F1 | 0.33 | Only 2 test rows |
| Validation accuracy | 0.00 | Only 1 validation row |
| Confidence | ~0.33 | Near-uniform probabilities |

The checkpoint predicted `fear` for both supplied test examples. These values demonstrate why dataset size and reliable evaluation design matter.

## Baseline Comparison

Run the included TF-IDF + Logistic Regression baseline:

```bash
python scripts/train_baseline.py --data data/emotion_dataset_full.csv
```

Then compare it against the attention model using accuracy, macro F1, and weighted F1. A neural network should not be presented as better unless it actually outperforms a credible baseline on the same split.

## Streamlit Demo

The app supports:

- manual text entry,
- preloaded emotion examples,
- predicted emotion and confidence,
- complete probability chart,
- attention-token table when available,
- CSV batch upload and automatic text-column selection,
- batch emotion-distribution chart,
- downloadable scored CSV,
- architecture, limitation, privacy, and responsible-use sections.

**Live demo:** `https://YOUR-APP-NAME.streamlit.app`  
**GitHub:** `https://github.com/USERNAME/bi-directional-lstm-projects`

Replace the placeholders after deployment.

## Local Setup

### 1. Open the project

```bash
cd bi-directional-lstm-projects/01-emotion-detection-bilstm-attention
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the supplied legacy demonstration

```bash
streamlit run app/streamlit_app.py
```

### 5. Train the real attention model

```bash
python scripts/train_model.py --data data/emotion_dataset_full.csv --epochs 15
```

The app automatically prefers the newly generated attention artifacts after restart.

### 6. Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Docker

```bash
docker build -t emotion-bilstm-attention .
docker run --rm -p 8501:8501 emotion-bilstm-attention
```

Open `http://localhost:8501`.

## Project Structure

```text
01-emotion-detection-bilstm-attention/
├── .streamlit/config.toml
├── app/streamlit_app.py
├── archive/original_emotion_detection_bilstm_notebook.ipynb
├── data/
│   ├── emotion_dataset.csv
│   ├── sample_emotion_data.csv
│   └── README_data.md
├── images/
├── models/
│   ├── legacy_emotion_bilstm_model.keras
│   ├── legacy_tokenizer_config.json
│   ├── legacy_label_mapping.json
│   ├── legacy_model_metadata.json
│   └── README.md
├── notebooks/emotion_detection_bilstm_attention.ipynb
├── outputs/
├── scripts/
│   ├── train_model.py
│   ├── train_baseline.py
│   ├── evaluate_model.py
│   ├── run_streamlit.py
│   └── validate_project.py
├── src/
│   ├── attention_layer.py
│   ├── baseline_model.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── emotion_prediction.py
│   ├── inference_pipeline.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   ├── sequence_generation.py
│   ├── text_preprocessing.py
│   ├── tokenizer_utils.py
│   └── visualization.py
├── tests/
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── README_HOSTING.md
└── README.md
```

## Hosting Recommendation

**Streamlit Community Cloud** is the recommended first option because the app is already Streamlit-native, deployment connects directly to GitHub, and the final app can be shared through a `streamlit.app` URL. See [`README_HOSTING.md`](./README_HOSTING.md).

## Limitations

- Emotion labels are dataset-specific and may not match a user's intended interpretation.
- Text can express multiple emotions while a softmax classifier returns one dominant class.
- Sarcasm, negation, slang, emojis, cultural context, and domain language remain difficult.
- Attention weights are useful diagnostics but are not guaranteed causal explanations.
- Confidence scores are not automatically calibrated probabilities.
- The bundled checkpoint is intentionally marked as a limited legacy artifact.

## Future Improvements

- Retrain on the full licensed dataset.
- Add probability calibration and confidence-threshold review.
- Compare against BiLSTM without attention and transformer baselines.
- Add pretrained embeddings or contextual encoders.
- Add multilingual and domain-specific evaluation.
- Add calibration plots, cross-validation, and experiment tracking.
- Package the model behind an API and add monitoring for drift and class imbalance.

## Skills Demonstrated

NLP preprocessing · sequence modeling · Bidirectional LSTM · temporal attention · multi-class classification · leakage prevention · class imbalance handling · model evaluation · error analysis · explainability · Streamlit deployment · Docker · CI testing · responsible AI communication

## Portfolio Positioning

**One-line description:** Built a production-structured BiLSTM-with-attention NLP pipeline for multi-class emotion detection, probability-based inference, error analysis, and Streamlit deployment.

**Pinned-repository description:** End-to-end emotion classification with TensorFlow/Keras BiLSTM attention, leakage-safe preprocessing, class-weighted training, rigorous evaluation, batch inference, and an interactive responsible-AI demo.

**Recommended screenshots:** single prediction result, probability chart, attention-token output, batch upload results, confusion matrix, training curves, and repository folder structure.
