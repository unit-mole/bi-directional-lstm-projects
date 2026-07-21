# Data documentation

## Training source used by the original notebook

The supplied notebook downloads **CoNLL-2003** through the Hugging Face dataset identifier `eriktks/conll2003` and uses the predefined train, validation, and test splits. Its token labels are:

`O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`, `B-MISC`, `I-MISC`.

The full benchmark is **not redistributed in this repository**. Newswire source text may have separate licensing conditions, so the training script downloads it from its public source at runtime. Review the upstream dataset card and source terms before redistribution or commercial use.

## Included safe samples

- `sample_ner_data.conll`: small synthetic token/tag examples
- `sample_ner_data.csv`: the same examples in token-per-row CSV format

These samples are intended for validation, demonstrations, and unit tests only. They are not large enough for meaningful model training.

## Expected local formats

CoNLL:

```text
Microsoft B-ORG
hired O
Priya B-PER
Shah I-PER

```

CSV:

```text
sentence_id,word,tag
1,Microsoft,B-ORG
1,hired,O
```

## Safety

Do not commit confidential case comments, customer data, resumes, medical records, legal text, serial numbers tied to individuals, or proprietary documents. Put private data under `data/raw/`, which is ignored by Git.
