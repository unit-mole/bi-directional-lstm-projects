# Emotion Detection using Bidirectional LSTM with Attention

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiLSTM%20%2B%20Attention-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-jujkaxt5ble9mx8f2zapjs.streamlit.app/)
[![CI](https://img.shields.io/badge/GitHub%20Actions-Project%2001-2088ff.svg)](../../actions)

Project 01 of the **Bi-Directional LSTM Projects** portfolio. This application classifies English text into six emotion classes—**anger, fear, joy, love, sadness, and surprise**—using a trained bidirectional LSTM and a learnable temporal-attention layer.

## Live application

**Streamlit:** https://bi-directional-lstm-projects-jujkaxt5ble9mx8f2zapjs.streamlit.app/

## Why this replacement was created

The earlier deployment used a three-class legacy checkpoint that returned nearly uniform probabilities around 33%. The positive and fear examples therefore produced almost identical outputs. This replacement removes that checkpoint from deployment and includes:

- a real six-class BiLSTM + temporal-attention architecture;
- a bundled trained PyTorch checkpoint;
- token-level attention visualization;
- single-text and batch-CSV inference;
- deterministic training, testing, and artifact validation;
- project-specific Streamlit and GitHub Actions configuration.

## Demonstration results

The bundled checkpoint was trained on **7,200 balanced, template-augmented educational examples** and evaluated on a stratified holdout of **1,080 rows**.

| Metric | Bundled holdout result |
|---|---:|
| Accuracy | 100.0% |
| Macro F1 | 100.0% |
| Weighted F1 | 100.0% |

> These scores measure a deterministic template-augmented demonstration dataset and must not be presented as real-world benchmark performance. For research-quality evaluation, retrain on a licensed natural-language corpus and report results on its untouched official test split.

## Expected smoke-test predictions

| Input | Expected class |
|---|---|
| `I am extremely happy and excited today.` | Joy |
| `I feel worried and anxious about the upcoming examination.` | Fear |
| `I feel lonely and heartbroken tonight.` | Sadness |
| `I am furious about the unfair decision.` | Anger |
| `I adore my family and feel so close to them.` | Love |
| `The unexpected announcement left me stunned.` | Surprise |

## Model architecture

```text
Cleaned tokens
    ↓
Train-only vocabulary encoding
    ↓
Embedding (96 dimensions)
    ↓
Bidirectional LSTM (64 units per direction)
    ↓
Temporal attention over sequence states
    ↓
Dense layer + dropout
    ↓
Six-class softmax output
```

The attention layer returns both the weighted context vector used for classification and normalized token weights shown in the Streamlit application.

## Repository structure

```text
01-emotion-detection-bilstm-attention/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── __init__.py
│   ├── requirements.txt
│   └── streamlit_app.py
├── archive/
│   └── legacy_checkpoint/
├── data/
│   ├── README_data.md
│   ├── emotion_dataset_full.csv
│   └── sample_emotion_data.csv
├── images/
├── models/
│   ├── emotion_bilstm_attention.pt
│   ├── label_mapping.json
│   ├── model_metadata.json
│   └── vocabulary.json
├── notebooks/
│   └── emotion_detection_bilstm_attention.ipynb
├── outputs/
│   ├── figures/
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── model_summary.txt
│   ├── test_predictions.csv
│   └── training_history.csv
├── scripts/
│   ├── evaluate_model.py
│   ├── generate_demo_dataset.py
│   ├── run_streamlit.py
│   ├── train_baseline.py
│   ├── train_model.py
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
├── README.md
├── README_HOSTING.md
├── requirements.txt
├── requirements-dev.txt
├── run_local.bat
├── run_local.sh
└── train_model.py
```

## Run locally on Windows

```cmd
run_local.bat
```

Manual setup:

```cmd
py -3.11 -m venv .venv
.venv\Scriptsctivate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Validate the packaged project

```cmd
python scripts/validate_project.py
python -m pytest -q
python scripts/evaluate_model.py
```

## Retrain the bundled architecture

Regenerate the bundled educational dataset:

```cmd
python scripts/generate_demo_dataset.py --output data/emotion_dataset_full.csv --rows-per-class 1200
```

Train:

```cmd
python scripts/train_model.py --data data/emotion_dataset_full.csv --epochs 12 --batch-size 64
```

To train on a stronger corpus, prepare a CSV with:

```csv
text,emotion
"I am delighted with the result",joy
"I feel anxious about tomorrow",fear
```

Then pass its path to the same training command. The loader also accepts `sentence`, `message`, `content`, or `comment` as the text-column name and `label`, `target`, or `class` as the target-column name.

## Streamlit deployment settings

| Field | Value |
|---|---|
| Repository | `unit-mole/bi-directional-lstm-projects` |
| Branch | `main` |
| Main file path | `01-emotion-detection-bilstm-attention/app/streamlit_app.py` |
| Python | `3.11` |
| Secrets | Leave blank |

## Outputs

![Class distribution](outputs/figures/class_distribution.png)

![Training accuracy](outputs/figures/training_accuracy.png)

![Confusion matrix](outputs/figures/confusion_matrix.png)

## Limitations and responsible use

- Emotion labels are simplified categories and do not capture mixed or culturally dependent expression.
- Sarcasm, negation, slang, and out-of-domain writing can reduce reliability.
- Attention weights are explanatory signals from this model, not proof of human reasoning.
- The bundled dataset is designed for an immediately runnable portfolio demonstration, not a real-world benchmark.
- Do not use the app for mental-health diagnosis, hiring, insurance, legal decisions, surveillance, or other consequential decisions.

## Dataset reference for stronger retraining

A suitable next-step corpus is the six-class DAIR.AI Emotion dataset described in `data/README_data.md`. Follow its usage restrictions and citation requirements before training or redistributing it.

## License

Project code is distributed under the repository MIT license. Third-party datasets remain governed by their own licenses and usage terms.
