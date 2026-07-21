# Original Attached Project Review

The archive preserves the user-supplied notebook, model, tokenizer, prediction analysis, and top-k evaluation. These files are retained for traceability and should not be presented as the final production-quality model.

Key observed results from the notebook:

- 8 resume records;
- 24 generated resume–job pairs;
- test set size of 4;
- test accuracy 0.75;
- positive-class precision 0.0;
- positive-class recall 0.0;
- positive-class F1 0.0;
- all four test probabilities near 0.49 and all predicted as no-match;
- Recall@5 of 0.8571 in a very small category-level ranking check;
- separate resume and job BiLSTM branches rather than true shared weights.

These findings motivated the corrected architecture and honest portfolio framing.
