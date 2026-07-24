# Bi-Directional LSTM Projects

A structured portfolio of six completed and deployed Bidirectional Long
Short-Term Memory projects covering text classification, healthcare NLP, named
entity recognition, semantic matching, resume–job comparison, code intelligence,
attention mechanisms, sequence-to-sequence generation, and interactive
Streamlit deployment.

**Portfolio status:** 6 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Repository:** [bi-directional-lstm-projects](https://github.com/unit-mole/bi-directional-lstm-projects)  
**Primary focus:** NLP · Sequence Modeling · Information Extraction · Semantic Matching · Code Intelligence

---

## Portfolio Objective

This repository demonstrates how Bidirectional LSTM architectures can be
applied to practical natural-language-processing, sequence-labeling,
semantic-similarity, information-extraction, and code-to-text problems.

Each project is developed as an end-to-end case study containing:

- a clearly defined problem statement;
- reproducible preprocessing and data-quality checks;
- a task-appropriate BiLSTM architecture;
- saved model, tokenizer, label, and metadata artifacts;
- reusable inference code;
- task-specific evaluation and error analysis;
- an interactive Streamlit demonstration;
- automated tests and project-specific GitHub Actions CI;
- local execution and deployment guidance;
- Docker support where applicable; and
- an honest discussion of model limitations, data quality, and responsible use.

The portfolio is designed to demonstrate skills relevant to Data Science,
Machine Learning, Applied AI, Natural Language Processing, Analytics
Engineering, Quality Analytics, Business Intelligence, and Code Intelligence
roles.

---

## Completed Projects

| No. | Project | Problem Type and Core Technique | Status |
|---:|---|---|---|
| 1 | [Emotion Detection](01-emotion-detection-bilstm-attention/) | Six-class emotion classification using a Bidirectional LSTM with temporal attention | [Live Demo](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/) |
| 2 | [Medical Text Classification](02-medical-text-classification-bilstm-attention/) | Five-class medical-specialty classification using a Bidirectional LSTM with temporal attention | [Live Demo](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/) |
| 3 | [Named Entity Recognition](03-named-entity-recognition-bilstm-crf/) | BIO sequence labeling using a BiLSTM baseline, constrained Viterbi decoding, and a true CRF training path | [Live Demo](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/) |
| 4 | [Question–Answer Matching](04-question-answer-matching-siamese-bilstm/) | Semantic text-pair classification and candidate ranking using a shared Siamese BiLSTM | [Live Demo](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/) |
| 5 | [Resume–Job Description Matching](05-resume-job-description-matching-siamese-bilstm/) | Shared Siamese BiLSTM matching supported by TF-IDF similarity and transparent skill coverage | [Live Demo](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/) |
| 6 | [Code Comment Generation](06-code-comment-generation-bilstm-attention/) | Python code-to-text generation using a BiLSTM encoder-decoder with a corrected Bahdanau-attention training path | [Live Demo](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/) |

---

## What the Portfolio Covers

The six projects are intentionally varied so that the repository demonstrates
multiple forms of sequence modeling rather than repeating one classification
pattern.

### Text Classification with Attention

- **Emotion Detection** classifies text into Anger, Fear, Joy, Love, Sadness,
  and Surprise while exposing temporal-attention information.
- **Medical Text Classification** classifies clinical-style text into five
  medical-specialty categories and provides confidence-aware predictions.

These projects demonstrate:

- text cleaning and tokenization;
- sequence padding and truncation;
- Bidirectional LSTM contextual encoding;
- temporal attention;
- multi-class softmax classification;
- class-probability interpretation;
- batch inference; and
- responsible communication of model confidence and limitations.

### Sequence Labeling and Information Extraction

- **Named Entity Recognition** assigns BIO tags to tokens and reconstructs
  Person, Organization, Location, and Miscellaneous entity spans.

This project demonstrates:

- token-level sequence classification;
- BIO tag validation;
- entity-span reconstruction;
- character-offset generation;
- Viterbi decoding;
- entity-level precision, recall, and F1;
- per-entity error analysis; and
- a clear distinction between a legacy softmax checkpoint and a true CRF
  training implementation.

### Semantic Matching and Retrieval

- **Question–Answer Matching** compares two texts using a shared Siamese BiLSTM
  and supports candidate-answer ranking.
- **Resume–Job Description Matching** compares resumes with job descriptions
  using shared neural encoders, TF-IDF similarity, and transparent skill
  coverage.

These projects demonstrate:

- shared-weight Siamese architectures;
- paired-text preprocessing;
- semantic-vector comparison;
- absolute-difference and element-wise-product features;
- binary probability scoring;
- ranking workflows;
- retrieval-oriented evaluation;
- transparent supporting signals; and
- privacy, fairness, and responsible-use documentation.

### Code Intelligence and Sequence Generation

- **Code Comment Generation** treats Python function documentation as a
  code-to-text sequence-generation problem.

This project demonstrates:

- syntax-aware Python preprocessing;
- source-docstring leakage control;
- separate source and target vocabularies;
- Bidirectional LSTM encoding;
- autoregressive LSTM decoding;
- teacher forcing;
- masked sequence loss;
- greedy and beam-search generation;
- BLEU, ROUGE, overlap, and length analysis; and
- a corrected Bahdanau-attention training path.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every project moves beyond notebook-only experimentation. The repository
demonstrates:

- problem definition;
- dataset auditing;
- reproducible preprocessing;
- training, validation, and test handling;
- model architecture design;
- task-specific evaluation;
- saved model and tokenizer artifacts;
- reusable prediction pipelines;
- single-record inference;
- batch inference;
- downloadable outputs;
- local execution;
- cloud deployment; and
- project-level documentation.

### Task-Appropriate Evaluation

The projects use metrics that match the underlying problem rather than relying
on accuracy alone.

Examples include:

- accuracy, macro F1, and weighted F1 for multi-class classification;
- probability distributions and confidence analysis;
- entity-level precision, recall, and F1 for named entity recognition;
- token-level BIO diagnostics;
- ROC-AUC and PR-AUC for binary semantic matching;
- threshold analysis;
- confusion matrices;
- ranking metrics such as Recall@K, MRR, and NDCG;
- BLEU, ROUGE, exact match, and token overlap for generation; and
- error-analysis tables and qualitative examples.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- modular source files instead of notebook-only logic;
- consistent preprocessing between training and inference;
- training-only vocabulary fitting in corrected pipelines;
- saved tokenizers, label mappings, metadata, and model artifacts;
- model-artifact validation scripts;
- unit tests for important preprocessing and inference paths;
- project-specific GitHub Actions workflows;
- nested `app/requirements.txt` files for Streamlit Community Cloud;
- Docker packaging;
- Git LFS handling for large Keras artifacts; and
- transparent legacy-artifact preservation.

### Interactive Application Development

Each project includes a deployed Streamlit application with task-appropriate
workflows.

Depending on the project, the applications provide:

- safe built-in examples;
- manual text entry;
- class probabilities;
- token- or attention-level interpretation;
- entity highlighting;
- candidate ranking;
- batch CSV upload;
- downloadable results;
- model architecture information;
- artifact-status reporting; and
- visible privacy, fairness, and responsible-use warnings.

### Honest Model Auditing

The repository does not hide limitations in the supplied checkpoints.

Examples include:

- identifying synthetic-data limitations;
- distinguishing package-validation metrics from real-world benchmarks;
- documenting undertrained or poorly calibrated checkpoints;
- identifying a legacy NER model that was not originally CRF-trained;
- identifying a code-generation checkpoint that did not contain the claimed
  attention layer;
- preserving original artifacts for traceability; and
- providing corrected training paths without falsely claiming that retraining
  has already occurred.

This demonstrates the ability to inspect, validate, document, and improve
existing machine-learning work rather than only presenting optimistic metrics.

---

## Project Summary

### 01 — Emotion Detection with BiLSTM and Attention

**Goal:** Classify text into six emotional categories.

**Core capabilities:**

- multi-class text classification;
- Bidirectional LSTM sequence encoding;
- temporal attention;
- class probabilities;
- attention-weighted token inspection;
- single-text and batch prediction;
- trained artifact persistence; and
- Streamlit deployment.

[Open project](01-emotion-detection-bilstm-attention/) ·
[Open live demo](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/)

### 02 — Medical Text Classification with BiLSTM and Attention

**Goal:** Classify clinical-style text into five medical-specialty categories.

**Core capabilities:**

- medical-text preprocessing;
- Bidirectional LSTM with temporal attention;
- confidence-aware five-class prediction;
- TF-IDF baseline comparison;
- batch classification;
- error analysis;
- privacy guidance; and
- visible medical-use limitations.

[Open project](02-medical-text-classification-bilstm-attention/) ·
[Open live demo](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/)

### 03 — Named Entity Recognition with BiLSTM and CRF-Aware Decoding

**Goal:** Extract Person, Organization, Location, and Miscellaneous entities
from unstructured text.

**Core capabilities:**

- BIO token tagging;
- entity-span reconstruction;
- constrained Viterbi decoding;
- token and entity-level evaluation;
- CoNLL validation;
- batch extraction;
- true linear-chain CRF training implementation; and
- transparent legacy-checkpoint auditing.

[Open project](03-named-entity-recognition-bilstm-crf/) ·
[Open live demo](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/)

### 04 — Question–Answer Matching with a Siamese BiLSTM

**Goal:** Estimate whether two texts express the same intent and rank candidate
answers.

**Core capabilities:**

- shared-weight Siamese encoders;
- semantic text-pair scoring;
- explicit pair-interaction features;
- threshold-aware classification;
- lexical-overlap diagnostics;
- candidate ranking;
- batch prediction; and
- honest undertraining disclosure.

[Open project](04-question-answer-matching-siamese-bilstm/) ·
[Open live demo](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/)

### 05 — Resume–Job Description Matching with a Siamese BiLSTM

**Goal:** Compare an anonymized resume with a job description and expose
supporting similarity signals.

**Core capabilities:**

- shared Siamese BiLSTM architecture;
- privacy-aware resume and job-text preprocessing;
- neural probability scoring;
- TF-IDF similarity;
- transparent skill-overlap analysis;
- blended fit scoring;
- batch inference;
- resume ranking; and
- fairness and employment-use limitations.

[Open project](05-resume-job-description-matching-siamese-bilstm/) ·
[Open live demo](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/)

### 06 — Code Comment Generation with BiLSTM and Attention

**Goal:** Generate a short natural-language description for a Python function.

**Core capabilities:**

- syntax-aware source-code preprocessing;
- docstring-leakage prevention;
- Bidirectional LSTM encoding;
- autoregressive LSTM decoding;
- greedy and beam search;
- transparent identifier baseline;
- BLEU and overlap evaluation;
- corrected Bahdanau-attention training architecture; and
- responsible code-privacy communication.

[Open project](06-code-comment-generation-bilstm-attention/) ·
[Open live demo](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/)

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this
structure:

```text
bi-directional-lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-emotion-detection-bilstm-attention.yml
│       ├── 02-medical-text-classification-bilstm-attention.yml
│       ├── 03-named-entity-recognition-bilstm-crf.yml
│       ├── 04-question-answer-matching-siamese-bilstm.yml
│       ├── 05-resume-job-description-matching-siamese-bilstm.yml
│       └── 06-code-comment-generation-bilstm-attention.yml
│
├── project-folder/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── archive/
│   ├── data/
│   │   └── README_data.md
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── LICENSE
│   ├── MODEL_CARD.md
│   ├── PROJECT_AUDIT.md
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements-dev.txt
│   └── requirements.txt
│
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

The exact files vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- deployable inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model limitations; and
- project-specific Streamlit demonstrations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Multi-class text classification | Emotion Detection and Medical Text Classification |
| Temporal attention | Emotion Detection and Medical Text Classification |
| Token-level sequence labeling | Named Entity Recognition |
| BIO decoding and entity reconstruction | Named Entity Recognition |
| Linear-chain CRF concepts | Named Entity Recognition corrected training path |
| Semantic text-pair matching | Question–Answer Matching |
| Siamese shared-weight architecture | Question–Answer Matching and Resume–Job Matching |
| Candidate ranking and retrieval | Question–Answer Matching and Resume–Job Matching |
| Code-to-text generation | Code Comment Generation |
| Encoder-decoder modeling | Code Comment Generation |
| Bahdanau attention | Code Comment Generation corrected training path |
| Teacher forcing | Code Comment Generation |
| Greedy and beam-search decoding | Code Comment Generation |
| Batch inference | All six projects |
| Model-artifact management | All six projects |
| Task-specific evaluation | Classification, NER, ranking, and generation projects |
| Interactive deployment | Six Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and six project-specific GitHub Actions workflows |
| Containerization | Project-level Dockerfiles |
| Responsible AI communication | Medical, hiring, semantic-matching, and code-generation projects |

---

## Technology Stack

`Python` · `TensorFlow` · `Keras` · `Bidirectional LSTM` ·
`Attention Mechanisms` · `Siamese Networks` · `CRF Concepts` ·
`scikit-learn` · `pandas` · `NumPy` · `Streamlit` · `Matplotlib` ·
`Plotly` · `NLTK` · `ROUGE` · `pytest` · `GitHub Actions` · `Docker` ·
`Git LFS`

---

## Core Skills Demonstrated

`Natural Language Processing` · `Sequence Modeling` ·
`Text Classification` · `Named Entity Recognition` · `BIO Tagging` ·
`Semantic Similarity` · `Information Retrieval` · `Code Intelligence` ·
`Sequence-to-Sequence Learning` · `Bidirectional LSTM` ·
`Temporal Attention` · `Bahdanau Attention` · `Siamese Neural Networks` ·
`Viterbi Decoding` · `CRF Training Concepts` · `Teacher Forcing` ·
`Beam Search` · `Tokenization` · `Model Evaluation` · `Error Analysis` ·
`Model Auditing` · `Artifact Persistence` · `Batch Inference` ·
`Streamlit Deployment` · `Testing` · `CI/CD` · `Docker` ·
`Responsible AI Communication`

---

## Local Setup

Clone the monorepo:

```bash
git clone https://github.com/unit-mole/bi-directional-lstm-projects.git
cd bi-directional-lstm-projects
```

Open the required project folder:

```bash
cd <project-folder>
```

Create a Python 3.11 virtual environment.

Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

Launch the project application:

```bash
python -m streamlit run app/streamlit_app.py
```

Each project README contains its own artifact-validation, training, evaluation,
Docker, and deployment instructions.

---

## Responsible Use

These projects are educational portfolio demonstrations.

The repository contains models involving medical text, resume matching,
semantic ranking, named entities, and generated code comments. Their outputs
must be reviewed by a qualified human before any real-world use.

In particular:

- medical-text outputs are not diagnoses or clinical decisions;
- resume–job scores must not determine hiring or rejection;
- semantic-match scores do not verify factual correctness;
- entity extraction can miss or misclassify sensitive information;
- generated code comments can be fluent but technically wrong; and
- private, confidential, medical, employment, customer, or proprietary text
  must not be submitted to the public applications.

Every project provides additional scope, privacy, fairness, and limitation
documentation.

---

## Current Limitations

Several projects use synthetic, small, or demonstration-focused datasets.

The repository therefore distinguishes between:

- proof that the complete pipeline works;
- package-validation or tiny-sample metrics;
- model behaviour on a limited checkpoint; and
- credible real-world generalization.

Project-specific limitations are documented in each project README,
`PROJECT_AUDIT.md`, and `MODEL_CARD.md` where available.

---

## Roadmap

- Retrain Project 03 using the included true CRF implementation and compare it
  with the legacy BiLSTM-softmax baseline.
- Retrain Project 06 using the corrected Bahdanau-attention architecture and a
  larger leakage-controlled dataset.
- Replace tiny demonstration datasets in Projects 02, 04, and 05 with larger,
  appropriately licensed datasets.
- Add probability calibration and confidence-based abstention where relevant.
- Add experiment tracking and formal model-version records.
- Add model cards and data cards across every project.
- Add automated deployment smoke tests for all six Streamlit applications.
- Compare recurrent architectures with transformer-based baselines.
- Add secure private-deployment patterns for medical, employment, and
  proprietary-code use cases.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI |
Natural Language Processing | Analytics Engineering | Quality Analytics
