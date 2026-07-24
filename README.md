# Bi-Directional LSTM Projects

A structured portfolio of six completed Bidirectional Long Short-Term Memory projects covering text classification, healthcare NLP, named entity recognition, semantic matching, resume–job comparison, code intelligence, attention mechanisms, sequence-to-sequence generation, and interactive Streamlit deployment.

**Portfolio status:** 6 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

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

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Emotion Detection](01-emotion-detection-bilstm-attention/) | Six-class emotion classification using a Bidirectional LSTM with temporal attention | [Live Demo](https://bi-directional-lstm-projects-hyy32ssueogmqtisqjxzbg.streamlit.app/) |
| 2 | [Medical Text Classification](02-medical-text-classification-bilstm-attention/) | Five-class medical-specialty classification using a Bidirectional LSTM with temporal attention | [Live Demo](https://bi-directional-lstm-projects-fvks4ksrhymci7vudgqq62.streamlit.app/) |
| 3 | [Named Entity Recognition](03-named-entity-recognition-bilstm-crf/) | BIO sequence labeling using a BiLSTM baseline, constrained Viterbi decoding, and a true CRF training path | [Live Demo](https://bi-directional-lstm-projects-qs6nxmeqrdur2reyifbxnh.streamlit.app/) |
| 4 | [Question–Answer Matching](04-question-answer-matching-siamese-bilstm/) | Semantic text-pair classification and candidate ranking using a shared Siamese BiLSTM | [Live Demo](https://bi-directional-lstm-projects-jvvrbxy7ukywyxzoqd4vnw.streamlit.app/) |
| 5 | [Resume–Job Description Matching](05-resume-job-description-matching-siamese-bilstm/) | Shared Siamese BiLSTM matching supported by TF-IDF similarity and transparent skill coverage | [Live Demo](https://bi-directional-lstm-projects-8xeeq2xagjbubsodntpurq.streamlit.app/) |
| 6 | [Code Comment Generation](06-code-comment-generation-bilstm-attention/) | Python code-to-text generation using a BiLSTM encoder-decoder with a corrected Bahdanau-attention training path | [Live Demo](https://bi-directional-lstm-projects-bd5p6qnsd6r5thune4tsk4.streamlit.app/) |

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

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Natural Language Processing | Analytics Engineering | Quality Analytics
