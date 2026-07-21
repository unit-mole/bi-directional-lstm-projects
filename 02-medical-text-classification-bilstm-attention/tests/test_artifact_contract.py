import json
import zipfile

from src.config import (
    DEFAULT_LABEL_MAPPING_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_PATH,
)


def test_required_artifacts_exist() -> None:
    for path in (
        DEFAULT_MODEL_PATH,
        DEFAULT_TOKENIZER_PATH,
        DEFAULT_LABEL_MAPPING_PATH,
        DEFAULT_METADATA_PATH,
    ):
        assert path.exists(), path


def test_keras_archive_contains_required_members() -> None:
    with zipfile.ZipFile(DEFAULT_MODEL_PATH) as archive:
        names = set(archive.namelist())
    assert {"metadata.json", "config.json", "model.weights.h5"}.issubset(names)


def test_tokenizer_json_is_supported_legacy_format() -> None:
    payload = json.loads(DEFAULT_TOKENIZER_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, (str, dict))
