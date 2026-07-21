# Dataset Guide

## Supplied file

`emotion_dataset.csv` and `sample_emotion_data.csv` currently contain the **10-row placeholder dataset** supplied with the notebook. Its columns are:

- `text` — short input sentence
- `emotion` — target emotion label

The six labels appearing in this placeholder are `joy`, `anger`, `fear`, `sadness`, `surprise`, and `calm`. Several classes have only one example, so this file is suitable for code and UI demonstrations—not credible model evaluation.

## Recommended full dataset workflow

1. Obtain the complete dataset from its original licensed source.
2. Review the dataset license and privacy terms before redistributing it.
3. Save the local file as `data/emotion_dataset_full.csv` or another explicit name.
4. Run:

```bash
python scripts/train_model.py --data data/emotion_dataset_full.csv
```

The loader dynamically detects common text and label columns. It does not hardcode a fixed label list.

## GitHub safety

Do not commit private messages, confidential customer feedback, personally identifiable data, or any full dataset whose license does not permit redistribution. The project `.gitignore` excludes `data/emotion_dataset_full.csv`, `data/raw/`, and `data/processed/` by default.
