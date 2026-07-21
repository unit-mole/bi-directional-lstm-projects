from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def token_f1(reference: str, prediction: str) -> float:
    ref = Counter(reference.split())
    pred = Counter(prediction.split())
    overlap = sum((ref & pred).values())
    if not ref or not pred or overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def sentence_bleu_score(reference: str, prediction: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        return float(sentence_bleu(
            [reference.split()], prediction.split(), smoothing_function=SmoothingFunction().method1
        ))
    except ImportError:
        return 0.0


def rouge_scores(reference: str, prediction: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1_f1": 0.0, "rouge2_f1": 0.0, "rougeL_f1": 0.0}
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(
        reference, prediction
    )
    return {
        "rouge1_f1": float(scores["rouge1"].fmeasure),
        "rouge2_f1": float(scores["rouge2"].fmeasure),
        "rougeL_f1": float(scores["rougeL"].fmeasure),
    }


def evaluate_pair(reference: str, prediction: str) -> dict[str, float]:
    return {
        "bleu": sentence_bleu_score(reference, prediction),
        "token_f1": token_f1(reference, prediction),
        "exact_match": float(reference.strip() == prediction.strip()),
        **rouge_scores(reference, prediction),
    }


def aggregate_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
