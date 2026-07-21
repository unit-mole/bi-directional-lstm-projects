# Model artifacts

## Supplied artifact

`legacy_bilstm_softmax_model.h5` is the exact model supplied with the original notebook. Inspection of its serialized architecture shows:

- input length: 124
- vocabulary size: 21,011
- embedding dimension: 64
- one BiLSTM with 64 units per direction
- dropout: 0.30
- TimeDistributed Dense(32, ReLU)
- TimeDistributed Dense(9, softmax)
- categorical cross-entropy loss

It does **not** contain a CRF layer or train with CRF likelihood. The filename and documentation therefore call it a legacy BiLSTM-softmax baseline.

## True CRF artifact

Run:

```bash
python scripts/train_model.py
```

This creates `ner_bilstm_crf.weights.h5`. The app always prefers these weights when present and reconstructs the architecture from `model_metadata.json`.

The repository uses a pure TensorFlow linear-chain CRF implementation instead of TensorFlow Addons. This reduces dependency fragility and keeps CRF transition scores, log-likelihood, and Viterbi decoding visible in the codebase.

## Mapping files

- `word_to_index.pkl`
- `index_to_word.pkl`
- `tag_to_index.pkl`
- `index_to_tag.pkl`
- `model_metadata.json`

Do not edit mappings independently of the model weights; training and inference must use identical IDs.
