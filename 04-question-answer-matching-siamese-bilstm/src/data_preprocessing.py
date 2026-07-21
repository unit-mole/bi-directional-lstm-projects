from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .text_preprocessing import clean_text

TEXT_A_ALIASES = ("question1", "q1", "question", "text_a", "sentence1", "query")
TEXT_B_ALIASES = ("question2", "q2", "answer", "candidate_answer", "text_b", "sentence2", "response")
LABEL_ALIASES = ("is_duplicate", "is_match", "label", "target", "relevance", "duplicate")


@dataclass(frozen=True)
class PairColumns:
    text_a: str
    text_b: str
    label: str | None = None


def _find_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def resolve_pair_columns(df: pd.DataFrame, *, require_label: bool = True) -> PairColumns:
    text_a = _find_column(df.columns, TEXT_A_ALIASES)
    text_b = _find_column(df.columns, TEXT_B_ALIASES)
    label = _find_column(df.columns, LABEL_ALIASES)
    if not text_a or not text_b:
        raise ValueError(
            "Could not identify both text columns. Supported examples include "
            "question1/question2, question/answer, text_a/text_b, and sentence1/sentence2."
        )
    if require_label and not label:
        raise ValueError("Could not identify the binary label column.")
    return PairColumns(text_a=text_a, text_b=text_b, label=label)


def normalize_binary_label(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing label")
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    positives = {"1", "1.0", "true", "yes", "match", "matched", "duplicate", "relevant", "similar"}
    negatives = {"0", "0.0", "false", "no", "no match", "unmatched", "not duplicate", "irrelevant", "dissimilar"}
    if text in positives:
        return 1
    if text in negatives:
        return 0
    raise ValueError(f"Unsupported binary label: {value!r}")


def load_pair_dataset(path: str | Path, *, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = resolve_pair_columns(df, require_label=require_label)
    rename_map = {columns.text_a: "text_a", columns.text_b: "text_b"}
    if columns.label:
        rename_map[columns.label] = "label"
    clean = df.rename(columns=rename_map).copy()
    required = ["text_a", "text_b"] + (["label"] if require_label else [])
    clean = clean.dropna(subset=required)
    clean["text_a"] = clean["text_a"].map(clean_text)
    clean["text_b"] = clean["text_b"].map(clean_text)
    clean = clean[(clean["text_a"] != "") & (clean["text_b"] != "")]
    if require_label:
        clean["label"] = clean["label"].map(normalize_binary_label).astype(int)
    clean["pair_key"] = clean.apply(
        lambda row: " ||| ".join(sorted((row["text_a"], row["text_b"]))), axis=1
    )
    clean = clean.drop_duplicates(subset=["pair_key"]).reset_index(drop=True)
    return clean


def standardize_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = resolve_pair_columns(df, require_label=False)
    result = df.rename(columns={columns.text_a: "text_a", columns.text_b: "text_b"}).copy()
    result = result.dropna(subset=["text_a", "text_b"]).reset_index(drop=True)
    return result
