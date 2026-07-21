"""Convert BIO/BILOU tag sequences into readable entity spans and HTML."""

from __future__ import annotations

import html
from collections import Counter
from typing import Sequence


def repair_bio_tags(tags: Sequence[str]) -> list[str]:
    """Repair illegal I-* transitions deterministically for robust display."""
    repaired: list[str] = []
    previous_type: str | None = None
    previous_prefix = "O"
    for tag in tags:
        if tag == "O":
            repaired.append(tag)
            previous_type, previous_prefix = None, "O"
            continue
        if "-" not in tag:
            repaired.append("O")
            previous_type, previous_prefix = None, "O"
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix == "I" and not (
            previous_prefix in {"B", "I"} and previous_type == entity_type
        ):
            tag = f"B-{entity_type}"
            prefix = "B"
        repaired.append(tag)
        previous_type, previous_prefix = entity_type, prefix
    return repaired


def extract_entities(
    tokens: Sequence[str],
    tags: Sequence[str],
    confidences: Sequence[float] | None = None,
    offsets: Sequence[tuple[int, int]] | None = None,
) -> list[dict[str, object]]:
    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must have the same length")
    if confidences is not None and len(confidences) != len(tokens):
        raise ValueError("confidences must align with tokens")
    if offsets is not None and len(offsets) != len(tokens):
        raise ValueError("offsets must align with tokens")

    tags = repair_bio_tags(tags)
    entities: list[dict[str, object]] = []
    start: int | None = None
    entity_type: str | None = None

    def close(end_exclusive: int) -> None:
        nonlocal start, entity_type
        if start is None or entity_type is None:
            return
        entity_tokens = list(tokens[start:end_exclusive])
        record: dict[str, object] = {
            "entity_text": " ".join(entity_tokens),
            "entity_type": entity_type,
            "start_token": start,
            "end_token": end_exclusive - 1,
        }
        if confidences is not None:
            values = confidences[start:end_exclusive]
            record["confidence"] = round(sum(values) / len(values), 4)
        if offsets is not None:
            record["char_start"] = offsets[start][0]
            record["char_end"] = offsets[end_exclusive - 1][1]
        entities.append(record)
        start, entity_type = None, None

    for index, tag in enumerate(tags):
        if tag == "O":
            close(index)
            continue
        prefix, current_type = tag.split("-", 1)
        if prefix in {"B", "U"}:
            close(index)
            start, entity_type = index, current_type
            if prefix == "U":
                close(index + 1)
        elif prefix in {"I", "L"}:
            if start is None or entity_type != current_type:
                close(index)
                start, entity_type = index, current_type
            if prefix == "L":
                close(index + 1)
    close(len(tokens))
    return entities


def entity_type_counts(entities: Sequence[dict[str, object]]) -> dict[str, int]:
    return dict(Counter(str(entity["entity_type"]) for entity in entities))


def highlighted_text_html(text: str, entities: Sequence[dict[str, object]]) -> str:
    """Safely render character-offset entities as highlighted HTML spans."""
    usable = [
        entity for entity in entities
        if "char_start" in entity and "char_end" in entity
    ]
    usable.sort(key=lambda item: int(item["char_start"]))
    cursor = 0
    parts: list[str] = []
    for entity in usable:
        start, end = int(entity["char_start"]), int(entity["char_end"])
        if start < cursor:
            continue
        parts.append(html.escape(text[cursor:start]))
        label = html.escape(str(entity["entity_type"]))
        value = html.escape(text[start:end])
        parts.append(
            f'<mark style="padding:0.15rem 0.3rem;border-radius:0.3rem;">'
            f'{value} <small style="font-weight:700;">{label}</small></mark>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")
