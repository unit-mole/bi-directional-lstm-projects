# Model Card — Code Comment Generation

## Intended use

Educational demonstration of Python function-to-comment generation using a bidirectional recurrent encoder and autoregressive decoder.

## Supplied checkpoint

The included checkpoint is a legacy BiLSTM encoder-decoder without attention. It was trained on 4,000 CodeSearchNet Python rows for five epochs.

## Measured performance

- BLEU: 0.0000 on 100 examples
- Exact match: 0.0000
- Mean reference-token overlap: 0.0878
- Validation loss: 4.0543
- Reported validation token accuracy: 0.4225, but unmasked and padding-inflated

## Limitations

The checkpoint can produce repetitive, generic, or incorrect text. It does not verify code behavior, security, performance, licensing, or edge cases. Training data may contain noisy documentation and repository-specific vocabulary.

## Responsible use

Do not submit proprietary or confidential source code to a public deployment. Do not merge generated documentation into production without developer review.

## Corrected model path

The repository contains a true Bahdanau-attention architecture and masked training pipeline. Retraining creates `code_comment_bilstm_attention_model.keras`, `encoder_model.keras`, and `decoder_model.keras`.
