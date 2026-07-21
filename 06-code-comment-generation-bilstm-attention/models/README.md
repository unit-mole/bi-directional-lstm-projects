# Model Artifacts

## Included checkpoint

- `code_comment_bilstm_seq2seq_model.keras`: supplied 70 MB Keras 3 checkpoint
- `code_tokenizer_config.json`: supplied source tokenizer
- `comment_tokenizer_config.json`: supplied target tokenizer
- `model_metadata.json`: audited configuration and measured results

The supplied checkpoint is reproducible but **does not include attention**. It is retained as a legacy baseline. The Streamlit app labels it honestly.

## Files produced after corrected retraining

```text
code_comment_bilstm_attention_model.keras
encoder_model.keras
decoder_model.keras
code_tokenizer_config.json
comment_tokenizer_config.json
model_metadata.json
```

Track `.keras` files with Git LFS. Never start model training during Streamlit app startup.
