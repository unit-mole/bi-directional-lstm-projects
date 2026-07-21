from __future__ import annotations

import random
from collections.abc import Iterable

import pandas as pd


def canonical_pair_key(text_a: str, text_b: str) -> str:
    return " ||| ".join(sorted((str(text_a).strip(), str(text_b).strip())))


def create_balanced_negative_pairs(
    positive_pairs: pd.DataFrame,
    *,
    text_a_column: str = "text_a",
    text_b_column: str = "text_b",
    random_state: int = 42,
) -> pd.DataFrame:
    """Create simple shuffled negatives while avoiding known positive unordered pairs."""
    rng = random.Random(random_state)
    positives = positive_pairs[[text_a_column, text_b_column]].dropna().drop_duplicates().copy()
    known = {
        canonical_pair_key(row[text_a_column], row[text_b_column])
        for _, row in positives.iterrows()
    }
    candidates = positives[text_b_column].tolist()
    negatives: list[dict[str, object]] = []
    for _, row in positives.iterrows():
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        for candidate in shuffled:
            key = canonical_pair_key(row[text_a_column], candidate)
            if key not in known and row[text_a_column] != candidate:
                negatives.append({"text_a": row[text_a_column], "text_b": candidate, "label": 0})
                break
    return pd.DataFrame(negatives)


def deduplicate_pairs(rows: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    output: list[tuple[str, str]] = []
    for text_a, text_b in rows:
        key = canonical_pair_key(text_a, text_b)
        if key not in seen:
            seen.add(key)
            output.append((text_a, text_b))
    return output
