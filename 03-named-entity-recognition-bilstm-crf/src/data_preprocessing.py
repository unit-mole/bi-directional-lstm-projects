"""Dataset loaders and validation for CoNLL, CSV, and Hugging Face NER data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class NERSentence:
    tokens: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("A sentence must contain at least one token.")
        if len(self.tokens) != len(self.tags):
            raise ValueError("Token and tag sequences must have identical lengths.")


def validate_bio_sequence(tags: Sequence[str]) -> list[str]:
    """Return human-readable BIO transition errors without mutating the sequence."""
    errors: list[str] = []
    previous_prefix = "O"
    previous_type: str | None = None
    for index, tag in enumerate(tags):
        if tag == "O":
            previous_prefix, previous_type = "O", None
            continue
        if "-" not in tag:
            errors.append(f"position {index}: malformed tag {tag!r}")
            previous_prefix, previous_type = tag, None
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            errors.append(f"position {index}: unsupported BIO prefix {prefix!r}")
        if prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_type == entity_type
        ):
            errors.append(
                f"position {index}: {tag} cannot follow "
                f"{tags[index - 1] if index else '<START>'}"
            )
        previous_prefix, previous_type = prefix, entity_type
    return errors


def load_conll(
    path: str | Path,
    token_column: int = 0,
    tag_column: int = -1,
    delimiter: str | None = None,
) -> list[NERSentence]:
    """Load blank-line-separated CoNLL token/tag data."""
    path = Path(path)
    sentences: list[NERSentence] = []
    tokens: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        nonlocal tokens, tags
        if tokens:
            sentences.append(NERSentence(tuple(tokens), tuple(tags)))
            tokens, tags = [], []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line.startswith("-DOCSTART-"):
                flush()
                continue
            parts = line.split(delimiter)
            try:
                token = parts[token_column].strip()
                tag = parts[tag_column].strip()
            except IndexError as exc:
                raise ValueError(
                    f"Line {line_number} does not contain the requested columns: {raw_line!r}"
                ) from exc
            if not token or not tag:
                raise ValueError(f"Line {line_number} has an empty token or tag.")
            tokens.append(token)
            tags.append(tag)
    flush()
    return sentences


def load_csv(
    path: str | Path,
    sentence_id_column: str = "sentence_id",
    token_column: str = "word",
    tag_column: str = "tag",
) -> list[NERSentence]:
    """Load a token-per-row CSV while preserving original row order."""
    frame = pd.read_csv(path)
    required = {sentence_id_column, token_column, tag_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[sentence_id_column] = frame[sentence_id_column].ffill()
    if frame[[sentence_id_column, token_column, tag_column]].isna().any().any():
        raise ValueError("Sentence IDs, tokens, and tags cannot contain unresolved missing values.")

    sentences: list[NERSentence] = []
    for _, group in frame.groupby(sentence_id_column, sort=False):
        tokens = tuple(group[token_column].astype(str).tolist())
        tags = tuple(group[tag_column].astype(str).tolist())
        sentences.append(NERSentence(tokens, tags))
    return sentences


def load_huggingface_conll2003(split: str) -> tuple[list[NERSentence], list[str]]:
    """Load the same public CoNLL-2003 source used by the original notebook."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to download CoNLL-2003.") from exc

    dataset = load_dataset("eriktks/conll2003", revision="convert/parquet", split=split)
    feature = dataset.features["ner_tags"].feature
    label_names = list(feature.names)
    sentences = [
        NERSentence(tuple(row["tokens"]), tuple(label_names[i] for i in row["ner_tags"]))
        for row in dataset
    ]
    return sentences, label_names


def dataset_statistics(sentences: Iterable[NERSentence]) -> dict[str, object]:
    sentences = list(sentences)
    token_count = sum(len(s.tokens) for s in sentences)
    tag_counts = Counter(tag for sentence in sentences for tag in sentence.tags)
    entity_counts = Counter(
        tag.split("-", 1)[1]
        for sentence in sentences
        for tag in sentence.tags
        if tag.startswith("B-")
    )
    lengths = [len(s.tokens) for s in sentences]
    return {
        "sentences": len(sentences),
        "tokens": token_count,
        "minimum_sentence_length": min(lengths, default=0),
        "maximum_sentence_length": max(lengths, default=0),
        "mean_sentence_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "tag_counts": dict(tag_counts),
        "entity_counts": dict(entity_counts),
        "bio_error_count": sum(len(validate_bio_sequence(s.tags)) for s in sentences),
    }


def save_conll(sentences: Iterable[NERSentence], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            for token, tag in zip(sentence.tokens, sentence.tags):
                handle.write(f"{token} {tag}\n")
            handle.write("\n")
