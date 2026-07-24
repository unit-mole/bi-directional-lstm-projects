# Dataset Guide

## Bundled dataset

`emotion_dataset_full.csv` contains 7,200 balanced, deterministic, template-augmented English examples across:

- anger
- fear
- joy
- love
- sadness
- surprise

It is included so the complete pipeline can be trained and deployed immediately. It is appropriate for application smoke testing and architecture demonstration, but not for real-world benchmark claims.

## Recommended research dataset

For stronger retraining, use the DAIR.AI Emotion dataset. Its split configuration contains 16,000 training, 2,000 validation, and 2,000 test examples and the same six labels. The dataset card states that it is intended for educational and research purposes.

Citation:

```bibtex
@inproceedings{saravia-etal-2018-carer,
  title = {CARER: Contextualized Affect Representations for Emotion Recognition},
  author = {Saravia, Elvis and Liu, Hsien-Chi Toby and Huang, Yen-Hao and Wu, Junlin and Chen, Yi-Shin},
  booktitle = {Proceedings of EMNLP 2018},
  year = {2018},
  pages = {3687--3697},
  doi = {10.18653/v1/D18-1404}
}
```

Before adding an external dataset to GitHub, confirm its license and redistribution terms.
