# Data Notes

`quora_question_pairs_sample.csv` is the 15-row file supplied with the original notebook. It is a synthetic Quora-style demonstration dataset with 10 positive and 5 negative pairs.

It is safe to commit because it is small and contains no private company data. It must not be described as the full Quora Question Pairs dataset or used to claim benchmark performance.

For credible duplicate-question modeling, download a licensed/public source separately and keep large raw files under `data/raw/`, which is ignored by Git. For genuine question-answer matching, use a dataset with explicit question, candidate-answer, and relevance labels.
