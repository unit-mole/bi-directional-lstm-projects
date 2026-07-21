# Bi-Directional LSTM Projects

A professional Bi-Directional LSTM portfolio featuring NLP classification, medical text classification, named entity recognition with CRF, Siamese semantic matching, resume-job matching, code comment generation, attention mechanisms, TensorFlow/Keras models, and interactive Streamlit demos.

## Career positioning

I am a Quality Data Scientist building an advanced sequence-modeling portfolio that connects production analytics, maintainable automation, natural-language processing, and applied AI. The repository demonstrates how bidirectional recurrent architectures can support classification, information extraction, semantic matching, and code-to-text generation workflows.

## Projects

| # | Project | Core technique | Status | Demo |
|---|---|---|---|---|
| 01 | [Emotion Detection](01-emotion-detection-bilstm-attention/) | BiLSTM + attention | Complete | `Add Streamlit URL` |
| 02 | [Medical Text Classification](02-medical-text-classification-bilstm-attention/) | BiLSTM + attention | Complete | `Add Streamlit URL` |
| 03 | [Named Entity Recognition](03-named-entity-recognition-bilstm-crf/) | BiLSTM-CRF | Complete | `Add Streamlit URL` |
| 04 | [Question-Answer Matching](04-question-answer-matching-siamese-bilstm/) | Siamese BiLSTM | Complete | `Add Streamlit URL` |
| 05 | [Resume-Job Description Matching](05-resume-job-description-matching-siamese-bilstm/) | Siamese BiLSTM | Complete | `Add Streamlit URL` |
| 06 | [Code Comment Generation](06-code-comment-generation-bilstm-attention/) | BiLSTM encoder-decoder + attention | Complete project scaffold; attention checkpoint should be retrained | `Add Streamlit URL` |

## Technology stack

Python, TensorFlow, Keras, scikit-learn, pandas, NumPy, Streamlit, Matplotlib, Plotly, NLTK, ROUGE evaluation, Docker, GitHub Actions, and Git LFS.

## Repository organization

```text
bi-directional-lstm-projects/
├── .github/workflows/
├── 01-emotion-detection-bilstm-attention/
├── 02-medical-text-classification-bilstm-attention/
├── 03-named-entity-recognition-bilstm-crf/
├── 04-question-answer-matching-siamese-bilstm/
├── 05-resume-job-description-matching-siamese-bilstm/
├── 06-code-comment-generation-bilstm-attention/
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

## Skills demonstrated

- Bidirectional sequence encoders and attention mechanisms
- Classification, sequence labeling, semantic similarity, and generation
- Reproducible preprocessing, evaluation, model artifact management, and inference
- Streamlit application development and responsible AI communication
- CI validation, Docker packaging, testing, documentation, and deployment planning

## Setup

Each project is independently runnable. Open its folder, create a virtual environment, install its `requirements.txt`, and follow the project README.

## Responsible use

These projects are educational portfolio demonstrations. Outputs must be reviewed by a qualified human before real-world use, especially for medical text, hiring, and code-generation scenarios.

## Roadmap

- Retrain project 06 with the included true attention architecture and a larger leakage-controlled dataset
- Add hosted demos and screenshots for all six projects
- Add experiment tracking and model cards across the monorepo
- Compare recurrent baselines with Transformer-based systems
