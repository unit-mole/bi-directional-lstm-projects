# Project Audit

## Executive finding

The supplied files form a working **proof-of-pipeline**, not a credible medical-text classifier. The project correctly demonstrates tokenization, padding, a Bidirectional LSTM, a custom temporal attention layer, softmax probabilities, artifact saving, and basic evaluation. However, the supplied dataset contains only **10 synthetic rows** across **5 classes**, so the model cannot learn reliable category boundaries.

## Files reviewed

- Original notebook: `Medical_Text_Classification_using_Bi_Directional_LSTM_COMPLETE_corrected.ipynb`
- Dataset: `medical_transcriptions.csv`
- Saved Keras model: `medical_text_bilstm_attention_model.keras`
- Tokenizer configuration
- Label mapping
- Prediction analysis CSV
- Training history CSV

The original files are preserved under `archive/` or normalized into the new folder structure.

## Actual task identified

The notebook classifies short clinical-style transcriptions into these medical specialties:

1. Cardiology
2. Gastroenterology
3. Neurology
4. Orthopedic
5. Radiology

Actual dataset columns:

- text: `transcription`
- target: `medical_specialty`

## Actual data profile

| Item | Finding |
|---|---:|
| Total rows | 10 |
| Classes | 5 |
| Rows per class | 2 |
| Missing text | 0 |
| Missing labels | 0 |
| Duplicate texts | 0 |
| Mean text length | 11.1 words |
| Maximum text length | 15 words |

This is a synthetic demonstration file, not the larger public Medical Transcriptions dataset often associated with this task.

## Original model architecture

```text
Input length 300
→ Embedding (91 × 128)
→ Bidirectional LSTM (64 units per direction, return_sequences=True)
→ Custom temporal attention
→ Dropout 0.30
→ Dense 128 ReLU
→ Dropout 0.20
→ Dense 5 Softmax
```

Total trainable parameters: **128,049**.

## Original split and results

Because the sample is too small for a normal 70/15/15 stratified split, the notebook adjusted the temporary split to 50%.

| Split | Rows |
|---|---:|
| Train | 5 |
| Validation | 2 |
| Test | 3 |

Reported holdout results:

| Metric | Value |
|---|---:|
| Accuracy | 0.3333 |
| Macro F1 | 0.1000 |
| Weighted F1 | 0.1667 |
| Weighted precision | 0.1111 |
| Weighted recall | 0.3333 |
| Test loss | 1.5904 |

The model predicted **Orthopedic for all three test rows**, including Cardiology and Neurology examples. Confidence values were approximately **20.6%**, which is close to a uniform five-class distribution.

## Major gaps in the original version

1. The Streamlit cell was only a placeholder and did not load or score the model.
2. The notebook labeled itself “deployment ready” despite lacking a functional deployment app.
3. The sample size was too small for performance claims.
4. The original attention class lacked an explicit serialization decorator and reusable module.
5. The tokenizer was saved as a JSON string nested inside a JSON file.
6. There was no model metadata contract linking preprocessing, sequence length, labels, and metrics.
7. There was no batch inference, downloadable output, or CSV column selection.
8. There were no automated tests, CI workflow, Docker setup, or artifact validation.
9. The original cleaner removed all punctuation, including potentially meaningful medical notation.
10. Privacy, medical disclaimer, and responsible-use communication were missing.
11. Baseline comparison and reproducible error-analysis files were absent.
12. Model performance limitations were not stated strongly enough.

## Professional conclusion

The supplied artifact is retained because it is useful for demonstrating model loading and deployment mechanics. It must be presented honestly as a small synthetic proof-of-concept. For a strong healthcare NLP portfolio project, retrain the modular pipeline on a substantially larger, appropriately licensed, de-identified dataset and report leakage-safe validation results.
