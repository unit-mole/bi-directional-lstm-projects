# Improvement Roadmap

## P0 — required before showcasing model quality

- Retrain the included true-attention architecture with at least 20,000 leakage-controlled examples
- Evaluate on the official CodeSearchNet validation/test splits
- Report masked token accuracy, BLEU, ROUGE-1/2/L, token F1, and qualitative examples
- Replace the legacy model metadata with corrected checkpoint metadata
- Capture a real attention heatmap from the corrected decoder

## P1 — modeling

- Add subword tokenization or SentencePiece for identifiers
- Compare additive and dot-product attention
- Add coverage penalty and trigram repetition blocking
- Tune beam width and length penalty on validation data
- Use pretrained code embeddings or a Transformer baseline for comparison

## P2 — engineering

- Add experiment tracking and dataset version metadata
- Add model quantization for faster Streamlit startup
- Add an artifact download script for hosted checkpoints
- Add integration tests that load the model in a TensorFlow-enabled CI job
