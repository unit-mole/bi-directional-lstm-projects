"""Train a true BiLSTM-CRF on the public CoNLL-2003 splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR, OUTPUT_DIR, TrainingConfig
from src.data_preprocessing import load_conll, load_huggingface_conll2003
from src.model_training import train_model
from src.visualization import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-conll", type=Path, help="Optional local training CoNLL file")
    parser.add_argument("--validation-conll", type=Path, help="Optional local validation CoNLL file")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=124)
    parser.add_argument("--max-vocab-size", type=int, default=30_000)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--lstm-units", type=int, default=128)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.train_conll) != bool(args.validation_conll):
        raise ValueError("Provide both --train-conll and --validation-conll, or neither.")
    if args.train_conll:
        train_sentences = load_conll(args.train_conll)
        validation_sentences = load_conll(args.validation_conll)
        tag_order = None
    else:
        train_sentences, tag_order = load_huggingface_conll2003("train")
        validation_sentences, _ = load_huggingface_conll2003("validation")

    config = TrainingConfig(
        seed=args.seed,
        max_vocab_size=args.max_vocab_size,
        max_sequence_length=args.max_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    model, history, word_to_index, tag_to_index = train_model(
        train_sentences,
        validation_sentences,
        MODEL_DIR,
        OUTPUT_DIR,
        config,
        tag_order=tag_order,
    )
    plot_training_history(history, OUTPUT_DIR / "training_curve.png")
    print(json.dumps({
        "weights": str(MODEL_DIR / "ner_bilstm_crf.weights.h5"),
        "vocabulary_size": len(word_to_index),
        "tag_count": len(tag_to_index),
        "epochs_completed": len(history.get("loss", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
