from __future__ import annotations

from pathlib import Path

from .inference_pipeline import QAMatcher


def load_default_matcher(project_root: str | Path) -> QAMatcher:
    return QAMatcher.from_artifacts(Path(project_root) / "models")
