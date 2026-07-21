# Original project audit

## What the supplied project actually does

The notebook downloads `eriktks/conll2003`, uses the CoNLL-2003 predefined splits, lowercases tokens, builds a 21,011-item vocabulary, pads all sequences to 124 tokens, and predicts nine BIO labels for `PER`, `ORG`, `LOC`, `MISC`, and `O`.

The saved model has 1,415,177 parameters and contains:

```text
Embedding(21011, 64, mask_zero=True)
→ Bidirectional LSTM(64 per direction)
→ Dropout(0.30)
→ TimeDistributed Dense(32, ReLU)
→ TimeDistributed Dense(9, softmax)
```

Its recorded test results are:

- SeqEval token accuracy: **0.9305**
- Entity-level micro F1: **0.6572**
- Entity precision: approximately **0.69**
- Entity recall: approximately **0.63**

## Critical finding

Despite the notebook title and narrative, the code does **not** implement a CRF:

- no trainable tag-transition matrix is present,
- no CRF log-likelihood is used,
- no CRF loss is compiled,
- each token is predicted independently with softmax,
- decoding is `argmax`, not learned Viterbi decoding.

Calling the supplied model “BiLSTM-CRF” without qualification would therefore be technically inaccurate.

## Portfolio correction

This repository:

1. preserves the original artifact as `legacy_bilstm_softmax_model.h5`,
2. exposes its real architecture and metrics,
3. adds a true linear-chain CRF loss and trainable transition matrix,
4. adds Viterbi decoding,
5. adds BIO validation and repair,
6. adds modular loaders, preprocessing, evaluation, error analysis, tests, CI, Docker, and Streamlit,
7. makes the app usable immediately with the legacy model while clearly labeling it,
8. automatically switches to true CRF weights after retraining.

## Additional technical weaknesses corrected

- Original vocabulary lowercases every token, losing capitalization cues useful for NER.
- Original tag padding uses `O`; evaluation correctly slices by true length, but training metrics can still be dominated by padded `O` positions depending on mask propagation.
- Hyperparameter search trained only two epochs on small slices, so its selected configuration should not be treated as a rigorous optimization study.
- Raw token accuracy is high partly because `O` is common; entity F1 is the more meaningful metric.
- The custom inference examples reveal OOV/generalization and invalid-boundary errors.
- The original notebook saves mappings but not complete metadata, reverse mappings, preprocessing configuration, or deployment code.
