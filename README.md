# Bi-Directional LSTM Projects

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-d00000.svg)](https://keras.io/)
[![NLP](https://img.shields.io/badge/NLP-Sequence%20Modeling-7c3aed.svg)](https://github.com/unit-mole/bi-directional-lstm-projects)
[![Streamlit](https://img.shields.io/badge/Streamlit-6%20Live%20Applications-ff4b4b.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Container%20Ready-2496ed.svg)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-Project--Specific%20CI-2088ff.svg)](https://github.com/unit-mole/bi-directional-lstm-projects/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A structured portfolio of six completed Bidirectional Long Short-Term Memory projects covering text classification, healthcare NLP, named entity recognition, semantic matching, resume–job comparison, code intelligence, attention mechanisms, sequence-to-sequence generation, and interactive Streamlit deployment.

**Portfolio status:** 6 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)  
**Deployment portfolio:** 6 Streamlit Community Cloud applications

---

## Portfolio Objective

This repository demonstrates how Bidirectional Long Short-Term Memory networks and related recurrent architectures can be applied to practical natural-language-processing, sequence-labeling, semantic-similarity, information-extraction, and code-to-text problems. Each project is developed as an end-to-end case study containing:

- a clearly defined business, analytical, or applied-AI problem;
- reproducible text preparation, tokenization, vocabulary construction, sequence padding, or syntax-aware preprocessing;
- leakage-aware training, validation, and test design;
- Bidirectional LSTM, attention, Siamese BiLSTM, encoder-decoder, Viterbi-decoding, or CRF-aware model development;
- task-appropriate baseline comparison and evaluation;
- saved tokenizer, label, metadata, weight, and model artifacts;
- modular and reusable classification, extraction, matching, ranking, generation, or inference code;
- an interactive Streamlit demonstration;
- automated tests and project-specific GitHub Actions CI;
- local execution, Docker, and deployment guidance;
- an honest discussion of assumptions, data quality, limitations, privacy, fairness, and responsible use.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Natural Language Processing, Analytics Engineering, Quality Analytics, Business Intelligence, Information Retrieval, and Code Intelligence roles.

---

## Completed Projects

| No. | Project | Sequence-Modeling Problem | Primary Deployment | Status |
|---:|---|---|---|---|
| 1 | [Emotion Detection](01-emotion-detection-bilstm-attention/) | Six-class emotion classification using a Bidirectional LSTM with temporal attention | Streamlit | [Live Demo](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/) |
| 2 | [Medical Text Classification](02-medical-text-classification-bilstm-attention/) | Five-class medical-specialty classification using a Bidirectional LSTM with temporal attention | Streamlit | [Live Demo](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/) |
| 3 | [Named Entity Recognition](03-named-entity-recognition-bilstm-crf/) | BIO sequence labeling using a BiLSTM baseline, constrained Viterbi decoding, and a true CRF training path | Streamlit | [Live Demo](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/) |
| 4 | [Question–Answer Matching](04-question-answer-matching-siamese-bilstm/) | Semantic text-pair classification and candidate ranking using a shared Siamese BiLSTM | Streamlit | [Live Demo](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/) |
| 5 | [Resume–Job Description Matching](05-resume-job-description-matching-siamese-bilstm/) | Shared Siamese BiLSTM matching supported by TF-IDF similarity and transparent skill coverage | Streamlit | [Live Demo](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/) |
| 6 | [Code Comment Generation](06-code-comment-generation-bilstm-attention/) | Python code-to-text generation using a BiLSTM encoder-decoder with a corrected Bahdanau-attention training path | Streamlit | [Live Demo](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/) |

---

## Portfolio at a Glance

| Portfolio Dimension | Evidence |
|---|---|
| Architecture family | Bidirectional LSTM and related recurrent architectures |
| Applied coverage | NLP, sequence labeling, semantic matching, retrieval, and code-to-text generation |
| End-to-end projects | 6 completed projects |
| Public applications | 6 Streamlit Community Cloud applications |
| Reproducibility | Project-level preprocessing, training, evaluation, and inference assets |
| Validation | Task-appropriate splits, baselines, metrics, diagnostics, and error analysis |
| Engineering | Modular code, tests, GitHub Actions, saved artifacts, and deployment guidance |
| Responsible use | Project-specific scope, limitations, privacy, fairness, and governance notes |

---

## What the Portfolio Covers

The six projects are intentionally varied so that the repository demonstrates multiple forms of Bidirectional LSTM-based sequence modeling rather than one repeated classification workflow.

### Text Classification with Attention

- **Emotion Detection** classifies text into Anger, Fear, Joy, Love, Sadness, and Surprise while exposing temporal-attention information.
- **Medical Text Classification** classifies clinical-style text into five medical-specialty categories and provides confidence-aware predictions.

These projects demonstrate text cleaning, tokenization, vocabulary control, sequence padding, Bidirectional LSTM contextual encoding, temporal attention, multi-class softmax classification, class-probability interpretation, baseline comparison, batch inference, and responsible communication of model confidence and limitations.

### Sequence Labeling and Information Extraction

- **Named Entity Recognition** assigns BIO tags to tokens and reconstructs Person, Organization, Location, and Miscellaneous entity spans.

This project demonstrates token-level sequence classification, BIO-tag validation, entity-span reconstruction, character-offset generation, constrained Viterbi decoding, entity-level precision, recall and F1, per-entity error analysis, and a transparent distinction between a legacy softmax checkpoint and a true CRF training implementation.

### Semantic Matching and Retrieval

- **Question–Answer Matching** compares two texts using a shared Siamese BiLSTM and supports candidate-answer ranking.
- **Resume–Job Description Matching** compares resumes with job descriptions using shared neural encoders, TF-IDF similarity, and transparent skill coverage.

These projects demonstrate paired-text preprocessing, shared-weight Siamese architectures, semantic-vector comparison, absolute-difference and element-wise-product interaction features, binary probability scoring, threshold analysis, ranking workflows, retrieval-oriented evaluation, transparent supporting signals, and privacy-, fairness-, and responsible-use documentation.

### Code Intelligence and Sequence Generation

- **Code Comment Generation** treats Python function documentation as a code-to-text sequence-generation problem.

This project demonstrates syntax-aware Python preprocessing, source-docstring leakage control, separate source and target vocabularies, Bidirectional LSTM encoding, autoregressive LSTM decoding, teacher forcing, masked sequence loss, greedy and beam-search generation, BLEU and ROUGE evaluation, overlap and length analysis, and a corrected Bahdanau-attention training path.

---

## Project Summaries

### 01 — Emotion Detection

[![Open Project 01](https://img.shields.io/badge/Open-Project%2001-2ea44f.svg)](01-emotion-detection-bilstm-attention/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/)

This project demonstrates **Six-class emotion classification using a Bidirectional LSTM with temporal attention** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](01-emotion-detection-bilstm-attention/) · [interactive application](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/)

---

### 02 — Medical Text Classification

[![Open Project 02](https://img.shields.io/badge/Open-Project%2002-2ea44f.svg)](02-medical-text-classification-bilstm-attention/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/)

This project demonstrates **Five-class medical-specialty classification using a Bidirectional LSTM with temporal attention** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](02-medical-text-classification-bilstm-attention/) · [interactive application](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/)

---

### 03 — Named Entity Recognition

[![Open Project 03](https://img.shields.io/badge/Open-Project%2003-2ea44f.svg)](03-named-entity-recognition-bilstm-crf/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/)

This project demonstrates **BIO sequence labeling using a BiLSTM baseline, constrained Viterbi decoding, and a true CRF training path** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](03-named-entity-recognition-bilstm-crf/) · [interactive application](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/)

---

### 04 — Question–Answer Matching

[![Open Project 04](https://img.shields.io/badge/Open-Project%2004-2ea44f.svg)](04-question-answer-matching-siamese-bilstm/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/)

This project demonstrates **Semantic text-pair classification and candidate ranking using a shared Siamese BiLSTM** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](04-question-answer-matching-siamese-bilstm/) · [interactive application](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/)

---

### 05 — Resume–Job Description Matching

[![Open Project 05](https://img.shields.io/badge/Open-Project%2005-2ea44f.svg)](05-resume-job-description-matching-siamese-bilstm/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/)

This project demonstrates **Shared Siamese BiLSTM matching supported by TF-IDF similarity and transparent skill coverage** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](05-resume-job-description-matching-siamese-bilstm/) · [interactive application](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/)

---

### 06 — Code Comment Generation

[![Open Project 06](https://img.shields.io/badge/Open-Project%2006-2ea44f.svg)](06-code-comment-generation-bilstm-attention/)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b.svg)](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/)

This project demonstrates **Python code-to-text generation using a BiLSTM encoder-decoder with a corrected Bahdanau-attention training path** through a reproducible recurrent-neural-network workflow. The project directory contains the task-specific data preparation, model development, evaluation evidence, reusable inference components, application code, tests, and responsible-use documentation.

**Project evidence:** [source and documentation](06-code-comment-generation-bilstm-attention/) · [interactive application](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/)

---

## Bidirectional LSTM Architecture Coverage

| Area | Demonstrated Through |
|---|---|
| Multi-class text classification | Emotion Detection and Medical Text Classification |
| Temporal attention | Emotion Detection and Medical Text Classification |
| Token-level sequence labeling | Named Entity Recognition |
| BIO decoding and entity reconstruction | Named Entity Recognition |
| Linear-chain CRF concepts | Named Entity Recognition corrected training path |
| Semantic text-pair matching | Question–Answer Matching |
| Resume–job semantic comparison | Resume–Job Description Matching |
| Siamese shared-weight architecture | Question–Answer Matching and Resume–Job Description Matching |
| Candidate ranking and retrieval | Question–Answer Matching and Resume–Job Description Matching |
| Code-to-text generation | Code Comment Generation |
| Encoder-decoder modeling | Code Comment Generation |
| Bahdanau attention | Code Comment Generation corrected training path |
| Teacher forcing | Code Comment Generation |
| Greedy and beam-search decoding | Code Comment Generation |
| Tokenization and vocabulary control | All six projects |
| Sequence padding and masking | Classification, NER, matching, and generation projects |
| Training-only preprocessing | Corrected tokenization, vocabulary, and threshold workflows |
| Classification thresholding | Question–Answer Matching and Resume–Job Description Matching |
| Baseline comparison | TF-IDF, lexical-overlap, identifier, and transparent heuristic baselines |
| Classification evaluation | Accuracy, macro F1, weighted F1, ROC-AUC, PR-AUC, and confusion matrices |
| Entity-level evaluation | Precision, recall, F1, BIO diagnostics, and per-entity analysis |
| Ranking evaluation | Recall@K, MRR, and NDCG |
| Generative NLP evaluation | BLEU, ROUGE, exact match, overlap, length, and qualitative analysis |
| Manual inference | Interactive Streamlit input workflows |
| Batch inference | CSV upload, sample scoring, ranking, and downloadable outputs |
| Model-artifact management | All six projects |
| Model auditing | Legacy-checkpoint inspection and corrected training paths |
| Model deployment | Six Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and six project-specific GitHub Actions workflows |
| Containerization | Project-level Dockerfiles |
| Large-file management | Git LFS for model artifacts |
| Responsible AI communication | Medical, hiring, semantic-matching, NER, and code-generation projects |

---

## Evaluation Coverage

The projects select evaluation methods according to the task rather than relying on a single universal metric. Classification, forecasting, anomaly detection, ranking, sequence labeling, representation learning, and generation require different evidence.

The projects use evaluation metrics that match the task rather than relying on one headline score.

Examples include:

- accuracy, macro F1, weighted F1, confusion matrices, and class-probability analysis for multi-class classification;
- entity-level precision, recall, and F1 together with token-level BIO diagnostics for named entity recognition;
- ROC-AUC, PR-AUC, threshold sensitivity, and confusion matrices for binary semantic matching;
- Recall@K, mean reciprocal rank, and normalized discounted cumulative gain for candidate ranking;
- BLEU, ROUGE, exact match, token overlap, length analysis, and qualitative review for code-comment generation;
- TF-IDF, lexical-overlap, identifier-based, or other transparent baselines to determine whether the neural model adds measurable value.

### Why multiple evaluation methods matter

- Accuracy alone can hide class-specific and minority-class weaknesses.
- Forecasting must preserve chronology and compare against transparent baselines.
- Ranking systems must be evaluated at the positions users actually inspect.
- Reconstruction error requires a documented threshold-selection strategy.
- Generated text requires both automated metrics and qualitative review.
- Confidence and similarity scores should not automatically be treated as calibrated probabilities.
- Negative results and rejected model candidates remain valuable engineering evidence.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business and analytical problem definition;
- reproducible text, token, label, pair, and source-code preparation;
- vocabulary, sequence, semantic-pair, and syntax-aware feature engineering;
- training, validation, and test separation;
- Bidirectional LSTM model development and task-specific evaluation;
- saved tokenizer, label, metadata, weight, and model artifacts;
- reusable classification, extraction, matching, ranking, and generation pipelines;
- manual, sample, and batch inference;
- downloadable classifications, entity spans, ranked candidates, match scores, and generated comments;
- local execution;
- cloud deployment.

### Sequence Modeling with Correct Validation

Natural-language and paired-sequence problems require careful validation and preprocessing. The repository emphasizes:

- training-only tokenizer and vocabulary fitting in corrected pipelines;
- consistent token, label, padding, and sequence order during training and inference;
- sample-, statement-, document-, pair-, or function-level splitting where appropriate;
- leakage control for source docstrings and duplicated text patterns;
- validation-based model, threshold, or stopping decisions;
- entity-level evaluation rather than token accuracy alone for named entity recognition;
- candidate-level and ranking-based evaluation for semantic retrieval;
- untouched final test evaluation where applicable;
- explicit documentation of synthetic-data, small-sample, checkpoint, calibration, and generalization risks.

### Model Evaluation Based on the Actual Problem

The projects use evaluation metrics that match the task rather than relying on one headline score.

Examples include:

- accuracy, macro F1, weighted F1, confusion matrices, and class-probability analysis for multi-class classification;
- entity-level precision, recall, and F1 together with token-level BIO diagnostics for named entity recognition;
- ROC-AUC, PR-AUC, threshold sensitivity, and confusion matrices for binary semantic matching;
- Recall@K, mean reciprocal rank, and normalized discounted cumulative gain for candidate ranking;
- BLEU, ROUGE, exact match, token overlap, length analysis, and qualitative review for code-comment generation;
- TF-IDF, lexical-overlap, identifier-based, or other transparent baselines to determine whether the neural model adds measurable value.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- modular source files rather than notebook-only logic;
- consistent preprocessing between training and prediction;
- saved tokenizers, vocabularies, label mappings, metadata, Keras models, weights, and portable artifacts;
- safe handling of empty text, malformed uploads, unknown tokens, invalid labels, incompatible shapes, and missing artifacts;
- pretrained application startup without automatic retraining;
- artifact-validation scripts and transparent checkpoint auditing;
- automated tests for important preprocessing and inference paths;
- project-specific GitHub Actions workflows;
- nested `app/requirements.txt` files for Streamlit Community Cloud;
- project-level Docker support;
- Git LFS handling for large model artifacts;
- GitHub-safe data, dependency, and artifact management.

### Business and Analytical Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- predicted emotion categories and class probabilities;
- medical-specialty predictions with confidence context;
- extracted named entities, BIO tags, spans, and character offsets;
- semantic match probabilities and decision thresholds;
- candidate-answer and resume rankings;
- TF-IDF similarity and transparent skill-coverage indicators;
- generated code comments using greedy or beam-search decoding;
- attention, token, overlap, or ranking diagnostics;
- model and baseline comparisons;
- error interpretations;
- batch summaries;
- downloadable scored datasets and generated outputs.

This demonstrates the ability to translate technical model outputs into information that can be understood by analysts, engineers, quality teams, operations teams, recruiters, developers, managers, and other business stakeholders.

### Responsible Model Communication

Each project documents its intended scope and limitations. The repository avoids presenting educational portfolio models as production-ready medical, hiring, semantic-retrieval, information-extraction, or code-intelligence systems without additional validation, governance, monitoring, security controls, domain expertise, privacy safeguards, fairness assessment, and human oversight.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this structure:

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
- deployable pretrained inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model assumptions and limitations;
- project-specific Streamlit demonstrations.

---

## Continuous Integration

The repository uses project-specific GitHub Actions workflows rather than one oversized workflow for the entire monorepo. Depending on the project, CI validates source syntax, imports, tests, model configuration, application entry points, artifact references, documentation links, oversized files, and accidental secret inclusion.

Project workflows keep validation focused on the folders that changed and help ensure that documented inference and deployment paths remain reproducible.

[![Open GitHub Actions](https://img.shields.io/badge/Open-GitHub%20Actions-2088ff?style=for-the-badge)](https://github.com/unit-mole/bi-directional-lstm-projects/actions)

---

## Deployment Directory

All 6 projects provide interactive Streamlit demonstrations. Deployment-specific entry points, requirements, configuration, and supporting artifacts are maintained inside the corresponding project directories.

| Project | Live Application |
|---|---|
| 01 — Emotion Detection | [Open Streamlit application](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/) |
| 02 — Medical Text Classification | [Open Streamlit application](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/) |
| 03 — Named Entity Recognition | [Open Streamlit application](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/) |
| 04 — Question–Answer Matching | [Open Streamlit application](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/) |
| 05 — Resume–Job Description Matching | [Open Streamlit application](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/) |
| 06 — Code Comment Generation | [Open Streamlit application](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/) |

---

## Run a Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/unit-mole/bi-directional-lstm-projects.git
cd bi-directional-lstm-projects
```

### 2. Enter a project

```bash
cd 01-emotion-detection-bilstm-attention
```

Replace the folder name with the project you want to run.

### 3. Create and activate a virtual environment

**Windows**

```bat
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Follow the project README

Each project contains task-specific instructions for training, evaluation, testing, local inference, and Streamlit execution. Use the selected project's `README.md` as the authoritative run guide.

---

## Responsible Use

This repository is intended for education, experimentation, technical demonstration, and portfolio presentation. Model outputs depend on the quality, representativeness, and licensing of the underlying data and may fail on inputs outside the evaluated distribution.

The applications must not be treated as authoritative medical, financial, hiring, safety-critical, operational, or other consequential decision systems. Important outputs require trusted data, independent validation, domain expertise, appropriate monitoring, and human oversight.

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
| Resume–job semantic comparison | Resume–Job Description Matching |
| Siamese shared-weight architecture | Question–Answer Matching and Resume–Job Description Matching |
| Candidate ranking and retrieval | Question–Answer Matching and Resume–Job Description Matching |
| Code-to-text generation | Code Comment Generation |
| Encoder-decoder modeling | Code Comment Generation |
| Bahdanau attention | Code Comment Generation corrected training path |
| Teacher forcing | Code Comment Generation |
| Greedy and beam-search decoding | Code Comment Generation |
| Tokenization and vocabulary control | All six projects |
| Sequence padding and masking | Classification, NER, matching, and generation projects |
| Training-only preprocessing | Corrected tokenization, vocabulary, and threshold workflows |
| Classification thresholding | Question–Answer Matching and Resume–Job Description Matching |
| Baseline comparison | TF-IDF, lexical-overlap, identifier, and transparent heuristic baselines |
| Classification evaluation | Accuracy, macro F1, weighted F1, ROC-AUC, PR-AUC, and confusion matrices |
| Entity-level evaluation | Precision, recall, F1, BIO diagnostics, and per-entity analysis |
| Ranking evaluation | Recall@K, MRR, and NDCG |
| Generative NLP evaluation | BLEU, ROUGE, exact match, overlap, length, and qualitative analysis |
| Manual inference | Interactive Streamlit input workflows |
| Batch inference | CSV upload, sample scoring, ranking, and downloadable outputs |
| Model-artifact management | All six projects |
| Model auditing | Legacy-checkpoint inspection and corrected training paths |
| Model deployment | Six Streamlit Community Cloud applications |
| Testing and CI/CD | pytest and six project-specific GitHub Actions workflows |
| Containerization | Project-level Dockerfiles |
| Large-file management | Git LFS for model artifacts |
| Responsible AI communication | Medical, hiring, semantic-matching, NER, and code-generation projects |

---

## Core Skills Demonstrated

`Long Short-Term Memory` · `LSTM` · `Bidirectional LSTM` · `BiLSTM` · `Siamese BiLSTM` · `Encoder-Decoder` · `Sequence-to-Sequence Learning` · `Attention Mechanisms` · `Temporal Attention` · `Bahdanau Attention` · `Natural Language Processing` · `Sequence Modeling` · `Text Classification` · `Emotion Detection` · `Medical Text Classification` · `Named Entity Recognition` · `BIO Tagging` · `Entity-Span Reconstruction` · `Viterbi Decoding` · `CRF Concepts` · `Semantic Similarity` · `Text-Pair Classification` · `Information Retrieval` · `Candidate Ranking` · `Resume–Job Matching` · `Code Intelligence` · `Code-to-Text Generation` · `Teacher Forcing` · `Greedy Decoding` · `Beam Search` · `Tokenization` · `Vocabulary Management` · `Sequence Padding` · `Masking` · `Embeddings` · `Leakage Prevention` · `Threshold Selection` · `Precision–Recall Analysis` · `Baseline Comparison` · `Classification Evaluation` · `Entity-Level Evaluation` · `Ranking Evaluation` · `Generative NLP Evaluation` · `Error Analysis` · `Model Auditing` · `Artifact Persistence` · `Batch Inference` · `Responsible AI Communication` · `Privacy-Aware Deployment` · `Fairness-Aware Modeling` · `TensorFlow` · `Keras` · `NumPy` · `pandas` · `scikit-learn` · `NLTK` · `ROUGE` · `Matplotlib` · `Plotly` · `Streamlit` · `pytest` · `GitHub Actions` · `CI/CD` · `Docker` · `Git LFS` · `Business Translation`

---

## Portfolio Positioning

**One-line description:** 6 end-to-end Bidirectional LSTM projects spanning NLP, sequence labeling, semantic matching, retrieval, and code-to-text generation, with public applications, reproducible evaluation, automated testing, and responsible-use documentation.

**Pinned repository description:** Professional Bidirectional LSTM portfolio featuring 6 deployed projects across NLP, sequence labeling, semantic matching, retrieval, and code-to-text generation—with task-appropriate evaluation, reusable inference, Streamlit applications, project-specific CI, and responsible AI communication.

This portfolio demonstrates the ability to move from sequential data and analytical objectives through preprocessing, architecture selection, validation, artifact management, inference, deployment, and stakeholder-facing communication.

---

## License and Third-Party Materials

The original source code and original documentation in this repository are licensed under the [MIT License](LICENSE).

Datasets, pretrained models, model weights, embeddings, images, text corpora, and other third-party assets used by the individual projects are not relicensed by this repository. They remain subject to the licenses, terms of use, attribution requirements, and usage restrictions established by their respective owners.

Before reusing any third-party material, review the corresponding project README, dataset documentation, model card, original source, and provider terms. Inclusion in this portfolio does not transfer ownership or grant additional usage rights beyond those provided by the original owner.

Unless explicitly stated otherwise, trained models, evaluation outputs, and generated artifacts are provided for educational, research, and portfolio-demonstration purposes. They are not guaranteed to be suitable for production, medical, financial, hiring, safety-critical, or other high-risk applications.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Natural Language Processing | Analytics Engineering | Quality Analytics
