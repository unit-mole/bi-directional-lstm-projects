# Model Card — Resume–Job Shared Siamese BiLSTM

## Intended use

Educational demonstration of paired-text modeling, semantic similarity, ranking, threshold tuning, and Streamlit deployment.

## Out-of-scope use

The model must not be used as the sole basis for hiring, rejection, promotion, compensation, immigration, legal, or employment decisions.

## Training data

Small synthetic positive/negative pairs derived from eight short example resumes and synthetic job descriptions across seven categories. This data does not represent real applicant diversity, writing styles, seniority, career transitions, or job markets.

## Architecture

A shared tokenizer and shared embedding/Bidirectional LSTM encoder process both inputs. Dense comparison layers use the two embeddings, absolute difference, element-wise product, and cosine similarity.

## Known limitations

Tiny synthetic data, repeated resumes across splits, limited skill taxonomy, simplified labels, potential bias, no calibration study on external data, and no subgroup fairness evaluation.

## Required safeguards

Human review, representative evaluation, protected-attribute exclusion, bias testing, privacy and retention controls, auditability, candidate notice, legal review, and continuous monitoring.
