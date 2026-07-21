"""End-to-end training utilities for the true BiLSTM-CRF model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .config import TrainingConfig
from .crf_layer import BiLSTMCRFTagger
from .data_preprocessing import NERSentence
from .ner_preprocessing import (
    build_tag_vocabulary,
    build_word_vocabulary,
    encode_sentences,
    pad_tag_sequences,
    pad_token_sequences,
    save_pickle,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_arrays(
    sentences: Sequence[NERSentence],
    word_to_index: dict[str, int],
    tag_to_index: dict[str, int],
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_ids, tag_ids = encode_sentences(
        sentences,
        word_to_index,
        tag_to_index,
        lowercase=config.lowercase_tokens,
    )
    x, lengths, _ = pad_token_sequences(
        token_ids,
        config.max_sequence_length,
        pad_value=word_to_index[config.pad_token],
    )
    y = pad_tag_sequences(
        tag_ids,
        config.max_sequence_length,
        pad_value=tag_to_index["O"],
    )
    return x, y, lengths


def build_model(
    vocabulary_size: int,
    number_of_tags: int,
    config: TrainingConfig,
) -> BiLSTMCRFTagger:
    model = BiLSTMCRFTagger(
        vocab_size=vocabulary_size,
        num_tags=number_of_tags,
        max_sequence_length=config.max_sequence_length,
        pad_token_id=0,
        embedding_dim=config.embedding_dim,
        lstm_units=config.lstm_units,
        dense_units=config.dense_units,
        dropout_rate=config.dropout_rate,
        name="bilstm_crf_tagger",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate))
    return model


def train_model(
    train_sentences: Sequence[NERSentence],
    validation_sentences: Sequence[NERSentence],
    model_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
    tag_order: Sequence[str] | None = None,
) -> tuple[BiLSTMCRFTagger, dict[str, list[float]], dict[str, int], dict[str, int]]:
    config = config or TrainingConfig()
    set_global_seed(config.seed)
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    word_to_index = build_word_vocabulary(
        train_sentences,
        max_size=config.max_vocab_size,
        min_frequency=config.min_token_frequency,
        lowercase=config.lowercase_tokens,
        pad_token=config.pad_token,
        unknown_token=config.unknown_token,
    )
    tag_to_index = build_tag_vocabulary(train_sentences, preferred_order=tag_order)
    if "O" not in tag_to_index:
        raise ValueError("The NER tag vocabulary must contain O.")

    x_train, y_train, _ = prepare_arrays(train_sentences, word_to_index, tag_to_index, config)
    x_validation, y_validation, _ = prepare_arrays(
        validation_sentences, word_to_index, tag_to_index, config
    )

    model = build_model(len(word_to_index), len(tag_to_index), config)
    model(np.zeros((1, config.max_sequence_length), dtype=np.int32), training=False)
    weights_path = model_dir / "ner_bilstm_crf.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    model.load_weights(weights_path)

    save_pickle(word_to_index, model_dir / "word_to_index.pkl")
    save_pickle(tag_to_index, model_dir / "tag_to_index.pkl")
    save_pickle({i: tag for tag, i in tag_to_index.items()}, model_dir / "index_to_tag.pkl")
    save_pickle({i: word for word, i in word_to_index.items()}, model_dir / "index_to_word.pkl")

    metadata_path = model_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({
        "vocab_size": len(word_to_index),
        "num_tags": len(tag_to_index),
        "max_sequence_length": config.max_sequence_length,
        "lowercase_tokens": config.lowercase_tokens,
        "labels": [tag for tag, _ in sorted(tag_to_index.items(), key=lambda item: item[1])],
        "trained_crf_config": {
            "embedding_dim": config.embedding_dim,
            "lstm_units": config.lstm_units,
            "dense_units": config.dense_units,
            "dropout_rate": config.dropout_rate,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs_requested": config.epochs,
            "seed": config.seed,
        },
        "true_crf_artifact": {
            "weights_filename": weights_path.name,
            "status": "trained",
            "implementation": "Pure TensorFlow linear-chain CRF loss and Viterbi decoding",
        },
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, history.history, word_to_index, tag_to_index
