# Named Entity Recognition using BiLSTM with CRF

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-red)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![CI](https://img.shields.io/badge/GitHub_Actions-CI-success)](../../actions)

An end-to-end **Named Entity Recognition** portfolio project that converts unstructured text into structured `PER`, `ORG`, `LOC`, and `MISC` entity spans using a Bidirectional LSTM emission network and linear-chain CRF sequence decoding.

> **Responsible use and privacy:** This project is for education and portfolio demonstration only. NER models can miss entities, misclassify types, or return incomplete spans. Do not upload private, sensitive, confidential, personal, medical, legal, or proprietary text. Predictions must not be used as guaranteed truth or as the sole basis for legal, medical, financial, hiring, compliance, surveillance, safety-critical, or other consequential decisions.

## Live demo

**Streamlit:** `Add your deployed URL here`

## Recruiter-ready summary

Built a modular BiLSTM sequence tagger with true CRF likelihood and Viterbi decoding, BIO-aware entity extraction, entity-level evaluation, error analysis, automated tests, Docker, GitHub Actions, and an interactive Streamlit application.

## Why this project matters

Named Entity Recognition is token-level sequence labeling. Given a sentence or document, the model assigns a tag to each token and combines BIO tags into readable spans such as:

```text
Microsoft → ORG
Priya Shah → PER
Seattle → LOC
```

The same information-extraction pattern can support quality analytics by extracting product names, defect symptoms, failure modes, root-cause terms, serial-number patterns, organizations, customer locations, and case metadata from unstructured quality comments. A production quality-domain solution would require a domain-specific label schema and properly governed training data.

## Honest audit of the supplied files

The attached notebook, mappings, and `.h5` model were reviewed before this repository was generated.

The original project uses CoNLL-2003 and produces a useful **BiLSTM token-classification baseline**, but its saved model is not a CRF model. The serialized architecture ends with `TimeDistributed(Dense(9, softmax))` and was trained with categorical cross-entropy. It has no trainable transition matrix or CRF log-likelihood.

| Supplied baseline result | Value |
|---|---:|
| SeqEval token accuracy | 0.9305 |
| Entity-level micro F1 | 0.6572 |
| Entity precision | ~0.69 |
| Entity recall | ~0.63 |

The original artifact is preserved as `models/legacy_bilstm_softmax_model.h5`. The app can use it immediately with transparent **BIO-constrained Viterbi post-processing**. The repository also adds a real CRF training path; after training, the app automatically prefers `models/ner_bilstm_crf.weights.h5`.

See [PROJECT_AUDIT.md](PROJECT_AUDIT.md) for the detailed findings.

## Dataset and labels

The original notebook loads `eriktks/conll2003` and uses the predefined CoNLL-2003 splits. The full newswire dataset is not redistributed here. A small synthetic sample is included for testing.

BIO labels:

```text
O
B-PER  I-PER
B-ORG  I-ORG
B-LOC  I-LOC
B-MISC I-MISC
```

`B-` begins an entity, `I-` continues the same entity, and `O` marks a non-entity token. The project preserves token order, punctuation, sentence boundaries, and token-tag alignment. It does not apply aggressive text cleaning.

## Architecture

```text
Token IDs
   ↓
Embedding
   ↓
Bidirectional LSTM
   ↓
Dropout
   ↓
TimeDistributed Dense projection
   ↓
Per-token emission scores
   ↓
Linear-chain CRF transition matrix
   ↓
Viterbi-decoded BIO sequence
   ↓
Human-readable entity spans
```

A BiLSTM learns left and right context. The CRF learns dependencies between neighboring output tags. For example, `I-PER` should normally follow `B-PER` or `I-PER`, not appear arbitrarily after `O`. Instead of choosing every token label independently, Viterbi decoding selects the highest-scoring complete sequence.

The CRF implementation is written directly in TensorFlow and includes:

- unary and transition sequence scores,
- forward-algorithm log normalization,
- negative log-likelihood training,
- masking based on true sequence length,
- trainable transition parameters,
- Viterbi decoding.

TensorFlow Addons is not required.

## Project workflow

1. Load CoNLL, CSV, or Hugging Face token/tag data.
2. Validate sentence grouping, token-tag alignment, and BIO transitions.
3. Build the vocabulary from training data only.
4. Map unseen tokens to `<UNK>` and padding to `<PAD>`.
5. Post-pad token and tag sequences consistently.
6. Train the BiLSTM emission network with CRF likelihood.
7. Decode test sequences using Viterbi.
8. Evaluate entity precision, recall, F1, per-type metrics, token metrics, and confusion patterns.
9. Convert BIO tags to spans with token indices, character offsets, and emission confidence.
10. Serve manual and batch predictions through Streamlit.

## Evaluation

Raw token accuracy can be misleading because `O` is often the majority label. The primary metric is **entity-level F1** from `seqeval`:

- Precision: how many predicted entity spans are correct.
- Recall: how many true entity spans are found.
- F1: harmonic balance of precision and recall.
- Per-type metrics: reveal whether `PER`, `ORG`, `LOC`, or `MISC` is difficult.

The evaluation script saves:

```text
outputs/entity_level_classification_report.csv
outputs/token_level_classification_report.csv
outputs/confusion_matrix.png
outputs/error_analysis.csv
outputs/model_metrics.json
```

The repository intentionally does not claim CRF results until CRF training and evaluation have actually been run.

## Streamlit demo features

- Manual text input and curated examples
- Highlighted entity spans
- Extracted entity table with token and character boundaries
- Token-level BIO tag table
- Entity-type distribution chart
- CSV batch inference with downloadable output
- CoNLL upload validation
- Model artifact details and limitations
- Responsible-use and privacy warning

The model is loaded once with `st.cache_resource`; training is never triggered by app startup.

## Local setup

Use Python 3.12.

```bash
cd bi-directional-lstm-projects
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r 03-named-entity-recognition-bilstm-crf/requirements-dev.txt
streamlit run 03-named-entity-recognition-bilstm-crf/app/streamlit_app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r 03-named-entity-recognition-bilstm-crf/requirements-dev.txt
streamlit run 03-named-entity-recognition-bilstm-crf/app/streamlit_app.py
```

## Train the true CRF model

From the project directory:

```bash
python scripts/train_model.py
```

Useful options:

```bash
python scripts/train_model.py \
  --epochs 15 \
  --batch-size 32 \
  --max-length 124 \
  --embedding-dim 100 \
  --lstm-units 128
```

This downloads CoNLL-2003, trains on the official training split, validates on the validation split, and saves:

```text
models/ner_bilstm_crf.weights.h5
models/word_to_index.pkl
models/tag_to_index.pkl
models/index_to_tag.pkl
models/model_metadata.json
outputs/training_history.csv
outputs/training_curve.png
```

## Evaluate

```bash
python scripts/evaluate_model.py
```

The evaluator uses the official test split and ignores padded positions.

## Tests and validation

```bash
python scripts/validate_project.py --skip-model-load
pytest -q
```

CI compiles the code, validates artifacts, imports the Streamlit app and inference pipeline, and runs lightweight tests. It does not retrain the model.

## Docker

```bash
docker build -t bilstm-crf-ner .
docker run --rm -p 8501:8501 bilstm-crf-ner
```

Open `http://localhost:8501`.

## Folder structure

```text
03-named-entity-recognition-bilstm-crf/
├── .streamlit/
├── app/
├── archive/
├── data/
├── images/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── Dockerfile
├── FILE_MANIFEST.csv
├── IMPROVEMENTS.md
├── MONOREPO_INTEGRATION.md
├── PROJECT_AUDIT.md
├── README.md
├── README_HOSTING.md
├── requirements.txt
├── requirements-dev.txt
├── run_local.bat
├── run_local.sh
└── train_model.py
```

## Error analysis focus

Review:

- missed entities,
- partial spans,
- wrong boundaries,
- `ORG` vs `LOC` confusion,
- `PER` vs `ORG` confusion,
- rare `MISC` entities,
- unknown names and domain shift,
- capitalization loss from the original lowercasing pipeline.

The supplied notebook's second custom example is a useful failure case: it misses `Barack Obama` and assigns inconsistent labels to `United Nations`.

## Limitations

- CoNLL-2003 supports only `PER`, `ORG`, `LOC`, and `MISC`.
- It is not a medical NER, resume NER, or quality-domain NER model.
- The original supplied vocabulary lowercases tokens and is word-level, so unseen names become `<UNK>`.
- Long documents are chunked and an entity may cross a chunk boundary.
- Emission confidence is not a calibrated probability of full entity correctness.
- The legacy model's high token accuracy should not be confused with high entity-span performance.

## Future improvements

- Retrain the true CRF model and publish verified comparison metrics.
- Compare BiLSTM-softmax versus BiLSTM-CRF under identical preprocessing.
- Add character-level CNN/LSTM features for unseen names.
- Preserve casing or add case-pattern features.
- Use pretrained embeddings or contextual encoders as a separate benchmark.
- Add domain-specific quality labels such as `PRODUCT`, `DEFECT`, `FAILURE_MODE`, `ROOT_CAUSE`, and `SERIAL_NUMBER` using governed internal data.
- Add confidence calibration and entity-level abstention.
- Add MLflow or experiment tracking.

## Portfolio descriptions

**One line:**

> End-to-end NER system using BiLSTM contextual encoding, CRF sequence decoding, BIO entity extraction, entity-level evaluation, and Streamlit deployment.

**Pinned-project description:**

> Built and deployed a modular BiLSTM-CRF sequence tagger for CoNLL-2003 NER with Viterbi decoding, entity-level F1 evaluation, error analysis, tests, Docker, and GitHub Actions.

## Skills demonstrated

Named Entity Recognition · token classification · BIO tagging · sequence modeling · Bidirectional LSTM · CRF likelihood · Viterbi decoding · entity-level metrics · error analysis · artifact management · Streamlit · Docker · CI/CD · responsible AI · information extraction

## Recommended screenshots

1. Streamlit input and highlighted entities
2. Extracted entity table
3. Token-to-BIO-tag table
4. Entity-type distribution chart
5. Training loss curve
6. Per-entity F1 chart
7. Confusion matrix after true CRF evaluation
8. GitHub folder structure and passing Actions workflow

## License

Code is MIT licensed. Dataset and third-party artifacts retain their original terms.
