from pathlib import Path

from src.inference_pipeline import CodeCommentInferencePipeline


def test_healthcheck_reports_missing_artifacts(tmp_path: Path):
    pipeline = CodeCommentInferencePipeline(tmp_path)
    ok, missing = pipeline.healthcheck()
    assert not ok
    assert "metadata" in missing


def test_committed_project_has_required_metadata_and_tokenizers():
    project_root = Path(__file__).resolve().parents[1]
    pipeline = CodeCommentInferencePipeline(project_root)
    ok, missing = pipeline.healthcheck(require_model=False)
    assert ok, missing
