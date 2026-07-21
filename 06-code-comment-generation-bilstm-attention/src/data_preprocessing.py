from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.code_preprocessing import CodePreprocessingOptions, preprocess_code
from src.comment_preprocessing import add_boundary_tokens, clean_comment, is_meaningful_comment

CODE_CANDIDATES = (
    "code_text", "func_code_string", "code", "source_code", "function_code", "function", "method", "snippet"
)
COMMENT_CANDIDATES = (
    "comment_text", "func_documentation_string", "docstring", "comment", "summary", "description", "target", "natural_language"
)


@dataclass(frozen=True)
class DatasetSchema:
    code_column: str
    comment_column: str
    language_column: str | None = None
    repository_column: str | None = None
    function_name_column: str | None = None


def _first_existing(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def infer_schema(df: pd.DataFrame) -> DatasetSchema:
    code_col = _first_existing(df.columns, CODE_CANDIDATES)
    comment_col = _first_existing(df.columns, COMMENT_CANDIDATES)
    if not code_col or not comment_col:
        raise ValueError(
            "Unable to identify code/comment columns. "
            f"Available columns: {list(df.columns)}"
        )
    return DatasetSchema(
        code_column=code_col,
        comment_column=comment_col,
        language_column=_first_existing(df.columns, ("language", "programming_language")),
        repository_column=_first_existing(df.columns, ("repository_name", "repository", "repo")),
        function_name_column=_first_existing(df.columns, ("func_name", "function_name", "name")),
    )


def load_tabular_dataset(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(source, lines=source.suffix.lower() == ".jsonl")
    raise ValueError(f"Unsupported dataset type: {source.suffix}")


def standardize_dataframe(
    df: pd.DataFrame,
    *,
    language: str = "python",
    preprocessing_mode: str = "semantic",
    max_code_tokens: int = 180,
    max_comment_tokens: int = 28,
) -> pd.DataFrame:
    schema = infer_schema(df)
    selected = pd.DataFrame({
        "code": df[schema.code_column],
        "comment": df[schema.comment_column],
    })
    selected["language"] = (
        df[schema.language_column].fillna(language).astype(str)
        if schema.language_column else language
    )
    if schema.repository_column:
        selected["repository"] = df[schema.repository_column].astype(str)
    if schema.function_name_column:
        selected["function_name"] = df[schema.function_name_column].astype(str)

    selected = selected.dropna(subset=["code", "comment"]).copy()
    selected["code"] = selected["code"].astype(str).str.strip()
    selected["comment"] = selected["comment"].astype(str).str.strip()
    selected = selected[(selected["code"] != "") & (selected["comment"] != "")]
    selected = selected.drop_duplicates(subset=["code", "comment"])

    options = CodePreprocessingOptions(max_tokens=max_code_tokens)
    selected["code_clean"] = selected["code"].map(
        lambda value: preprocess_code(value, options, mode=preprocessing_mode)
    )
    selected["comment_clean"] = selected["comment"].map(
        lambda value: clean_comment(value, max_tokens=max_comment_tokens)
    )
    selected = selected[selected["comment_clean"].map(is_meaningful_comment)]
    selected = selected[selected["code_clean"].str.split().str.len().between(3, max_code_tokens)]
    selected["comment_sequence"] = selected["comment_clean"].map(
        lambda value: add_boundary_tokens(value, max_content_tokens=max_comment_tokens)
    )
    return selected.reset_index(drop=True)


def load_codesearchnet_splits(
    *,
    language: str = "python",
    max_train_samples: int = 20_000,
    max_validation_samples: int = 2_000,
    max_test_samples: int = 2_000,
    preprocessing_mode: str = "semantic",
) -> dict[str, pd.DataFrame]:
    """Download CodeSearchNet lazily and preserve its official split boundaries."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the development requirements to download CodeSearchNet.") from exc

    raw = load_dataset("code_search_net", language, trust_remote_code=False)
    limits = {
        "train": max_train_samples,
        "validation": max_validation_samples,
        "test": max_test_samples,
    }
    output: dict[str, pd.DataFrame] = {}
    for split, limit in limits.items():
        source = raw[split].select(range(min(limit, len(raw[split])))).to_pandas()
        output[split] = standardize_dataframe(
            source,
            language=language,
            preprocessing_mode=preprocessing_mode,
        )
    return output
