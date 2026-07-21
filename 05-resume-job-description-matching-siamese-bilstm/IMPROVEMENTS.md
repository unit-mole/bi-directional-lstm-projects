# Improvements Made from the Attached Notebook

## Audit findings

The attached notebook is a useful proof of concept, but it is not yet a reliable Siamese matching project:

1. The source dataset contains only eight resumes and seven categories.
2. It creates 24 pairs: 8 positive and 16 negative.
3. The function named `build_shared_encoder` is called twice, producing separate embedding and BiLSTM layers. The resulting model is dual-branch, but its weights are not shared; therefore it is not a true Siamese encoder.
4. Resume and job maximum lengths are 300 and 220 although observed sequences are only about 13–20 tokens.
5. The saved model predicts every four-row test example as class 0. Accuracy is 0.75 only because three of four examples are negative; positive-class precision, recall, and F1 are all 0.0.
6. The reported Recall@5 is 0.8571 on seven category queries and eight candidate resumes, while the displayed data-science example ranks unrelated categories above data science.
7. The tokenizer is saved as a double-encoded JSON string.
8. The Streamlit cell is only a placeholder and does not load artifacts or perform inference.
9. The notebook states that the project is deployment-ready even though the app and production inference pipeline are missing.

## Implemented corrections

- Built a genuinely shared embedding + Bidirectional LSTM encoder and reused the same encoder model for both texts.
- Added transparent comparison features: vectors, absolute difference, element-wise product, and cosine similarity.
- Added a train-only tokenizer, OOV handling, fixed post-padding, saved metadata, threshold tuning, and artifact validation.
- Added synthetic, clearly documented, portfolio-safe demonstration job descriptions and balanced pairs.
- Added precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, baseline comparison, and ranking outputs.
- Added PII masking, fairness/privacy warnings, skill overlap explanations, and honest limitations.
- Added a complete Streamlit app with manual scoring, batch CSV inference, downloads, and resume ranking.
- Added modular source files, tests, Docker, local runners, hosting instructions, and lightweight GitHub Actions CI.
- Preserved the original notebook, model, tokenizer, and output CSVs under `archive/` for auditability.
