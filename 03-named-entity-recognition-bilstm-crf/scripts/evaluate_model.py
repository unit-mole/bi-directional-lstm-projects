"""Evaluate trained CRF weights on CoNLL-2003 or a local CoNLL test file."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, OUTPUT_DIR, TrainingConfig
from src.crf_layer import BiLSTMCRFTagger, viterbi_decode_numpy
from src.data_preprocessing import load_conll, load_huggingface_conll2003
from src.model_evaluation import build_error_analysis, evaluate_sequences
from src.model_training import prepare_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-conll", type=Path)
    args = parser.parse_args()

    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    with (MODEL_DIR / "word_to_index.pkl").open("rb") as handle:
        word_to_index = pickle.load(handle)
    with (MODEL_DIR / "tag_to_index.pkl").open("rb") as handle:
        tag_to_index = pickle.load(handle)
    index_to_tag = {index: tag for tag, index in tag_to_index.items()}

    test_sentences = load_conll(args.test_conll) if args.test_conll else load_huggingface_conll2003("test")[0]
    crf_config = metadata.get("trained_crf_config")
    weights_path = MODEL_DIR / metadata["true_crf_artifact"]["weights_filename"]
    if not crf_config or not weights_path.exists():
        raise FileNotFoundError("Train the true CRF model before running this evaluator.")

    config = TrainingConfig(
        max_sequence_length=int(metadata["max_sequence_length"]),
        embedding_dim=int(crf_config["embedding_dim"]),
        lstm_units=int(crf_config["lstm_units"]),
        dense_units=int(crf_config["dense_units"]),
        dropout_rate=float(crf_config["dropout_rate"]),
        lowercase_tokens=bool(metadata["lowercase_tokens"]),
    )
    x_test, y_test, lengths = prepare_arrays(test_sentences, word_to_index, tag_to_index, config)
    model = BiLSTMCRFTagger(
        vocab_size=len(word_to_index), num_tags=len(tag_to_index),
        max_sequence_length=config.max_sequence_length,
        embedding_dim=config.embedding_dim, lstm_units=config.lstm_units,
        dense_units=config.dense_units, dropout_rate=config.dropout_rate,
    )
    model(np.zeros((1, config.max_sequence_length), dtype=np.int32), training=False)
    model.load_weights(weights_path)
    emissions = model(x_test, training=False).numpy()
    predictions = viterbi_decode_numpy(emissions, model.transition_params.numpy(), lengths)
    true_tags = [[index_to_tag[int(i)] for i in row[:length]] for row, length in zip(y_test, lengths)]
    predicted_tags = [[index_to_tag[int(i)] for i in row] for row in predictions]
    metrics = evaluate_sequences(true_tags, predicted_tags, OUTPUT_DIR)
    build_error_analysis(
        [sentence.tokens[: int(length)] for sentence, length in zip(test_sentences, lengths)],
        true_tags,
        predicted_tags,
        OUTPUT_DIR / "error_analysis.csv",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
