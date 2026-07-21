# Data Guide

## Training source

The attached notebook used the **Python subset of CodeSearchNet**, specifically the fields `func_code_string`, `func_documentation_string`, `language`, `repository_name`, `func_name`, and related repository metadata. The original run selected the first 4,000 rows from the CodeSearchNet training split.

## Public repository policy

The complete CodeSearchNet download and the original raw prediction export are not included as public training data. Upstream repositories may use different licenses. Review source licenses before redistributing code. The committed `sample_code_comment_pairs.csv` and `sample_code_snippets.json` contain synthetic, non-proprietary examples for tests and the public demo.

## Leakage control

The original `func_code_string` values contain the target docstring. The corrected semantic preprocessing removes function/class/module docstrings from the source before training so the model cannot simply read the answer.

## Downloading the dataset

Install `requirements-dev.txt`, then run:

```bash
python scripts/train_model.py --train-samples 20000 --validation-samples 2000 --test-samples 2000
```

The training script downloads CodeSearchNet through Hugging Face Datasets and preserves the official train/validation/test split boundaries.

## Never upload

Do not upload company code, credentials, API keys, private repository snippets, internal endpoints, customer data, or copyrighted code that you are not allowed to redistribute.
