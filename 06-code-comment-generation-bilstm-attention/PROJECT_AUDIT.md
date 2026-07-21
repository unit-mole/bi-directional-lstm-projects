# Technical Audit of the Supplied Project

## What the supplied files actually do

The notebook uses the Python configuration of CodeSearchNet and maps:

- `func_code_string` → source sequence
- `func_documentation_string` → target comment/docstring
- `language` → Python metadata

It fits separate Keras tokenizers over 4,000 rows, pads code to 180 tokens and comments to 30 tokens, trains a 6,049,727-parameter BiLSTM encoder/LSTM decoder for five epochs, creates greedy and beam decoders, and evaluates 100 generated comments.

## Critical findings

1. **No attention layer exists in the supplied checkpoint.** The encoder returns only its final forward/backward states. The decoder never attends over encoder time-step outputs.
2. **Target leakage risk is present.** `func_code_string` contains the function docstring, which is also used as the target.
3. **Programming operators are lost.** The code tokenizer uses Keras' default filters, which remove punctuation/operators such as `+`, `-`, `*`, `/`, parentheses, and comparison symbols.
4. **Padding is counted as accuracy.** The model uses ordinary sparse categorical cross-entropy and accuracy without a padding mask. The reported 42.25% validation token accuracy is therefore inflated.
5. **Evaluation is weak.** BLEU is 0.0, exact match is 0.0, and average reference-token overlap is 0.0878 on 100 examples.
6. **Decoding needs correction.** The original beam search does not stop beams at `<end>` and does not apply length normalization.
7. **The Streamlit cell is only a placeholder.** It does not load artifacts or run inference.
8. **The experiment is undersized.** It uses 4,000 examples and five epochs from only the training split.

## Improvements implemented

- Added a true Bahdanau attention layer over encoder outputs
- Added docstring stripping and comment removal to reduce leakage
- Added lexical Python tokenization that preserves operators and keywords
- Added separate source/target tokenizers with explicit OOV and boundary tokens
- Added masked loss and masked token accuracy
- Preserved official CodeSearchNet split boundaries in the retraining script
- Added corrected greedy and beam search with `<end>` stopping and length normalization
- Added encoder/decoder inference model export
- Added BLEU, ROUGE, token F1, exact match, and qualitative error-analysis utilities
- Added a complete Streamlit interface, safe examples, batch mode, baseline comparison, and responsible-use warnings
- Added tests, CI, Docker, Git LFS configuration, hosting documentation, model card, and recruiter-facing README

## Honest checkpoint status

The supplied 70 MB model is retained as a reproducibility baseline. It is not relabeled as an attention model. Run `python scripts/train_model.py` to generate the corrected attention checkpoint before claiming attention-based results.
