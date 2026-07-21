# Project Audit

## What was working

- Correct use of a shared Siamese encoder.
- Explicit Bidirectional LSTM sequence modeling.
- Useful absolute-difference and element-wise-product interactions.
- Saved Keras model, tokenizer, training history, and prediction analysis.
- Basic train/validation/test split and early stopping.

## Critical weaknesses

- Fifteen rows are far too few for a neural semantic model.
- The tokenizer was fitted on the complete dataset before splitting.
- A three-row test set cannot support meaningful metrics.
- The model predicts nearly the same probability for every example.
- The negative class received zero recall in the saved test output.
- The original deployment code did not load or score the model.
- The title implied question-answer matching while the labels represented duplicate questions.

## Resolution

The rebuilt project adds honest task framing, a functioning app, reusable code, safer training behavior, broader evaluation, artifact metadata, tests, CI, Docker, hosting instructions, and a clear retraining path.
