from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .data_preprocessing import load_pair_dataset
from .model_evaluation import evaluate_probabilities, tune_threshold
from .sequence_generation import texts_to_padded_sequences
from .siamese_model import build_siamese_bilstm
from .tokenizer_utils import build_tokenizer, save_tokenizer


def train(
    data_path: str | Path,
    output_model_dir: str | Path,
    output_dir: str | Path,
    *,
    max_vocabulary_size: int = 40000,
    max_sequence_length: int = 40,
    embedding_dimension: int = 128,
    bilstm_units: int = 64,
    epochs: int = 20,
    batch_size: int = 64,
    random_state: int = 42,
) -> dict[str, object]:
    import tensorflow as tf

    df = load_pair_dataset(data_path, require_label=True)
    if len(df) < 100:
        raise ValueError(
            "Refusing to present a tiny dataset as a trained portfolio model. "
            "Provide at least 100 labelled pairs; several thousand are recommended."
        )

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df["label"]
    )
    validation_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_df["label"]
    )

    # Fit only on training text to avoid vocabulary leakage.
    tokenizer = build_tokenizer(
        train_df["text_a"].tolist() + train_df["text_b"].tolist(),
        num_words=max_vocabulary_size,
    )
    effective_vocab_size = min(max_vocabulary_size, len(tokenizer.word_index) + 1)

    def arrays(frame: pd.DataFrame):
        a = texts_to_padded_sequences(tokenizer, frame["text_a"].tolist(), max_length=max_sequence_length)
        b = texts_to_padded_sequences(tokenizer, frame["text_b"].tolist(), max_length=max_sequence_length)
        return a, b, frame["label"].to_numpy(dtype=int)

    train_a, train_b, train_y = arrays(train_df)
    val_a, val_b, val_y = arrays(validation_df)
    test_a, test_b, test_y = arrays(test_df)

    model = build_siamese_bilstm(
        vocab_size=effective_vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dimension=embedding_dimension,
        bilstm_units=bilstm_units,
    )
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_y)
    class_weights = {int(label): float(weight) for label, weight in zip(classes, weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-5),
    ]
    history = model.fit(
        [train_a, train_b],
        train_y,
        validation_data=([val_a, val_b], val_y),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_probability = model.predict([val_a, val_b], verbose=0).reshape(-1)
    threshold, _ = tune_threshold(val_y, val_probability)
    test_probability = model.predict([test_a, test_b], verbose=0).reshape(-1)
    metrics = evaluate_probabilities(test_y, test_probability, threshold=threshold)

    model_dir = Path(output_model_dir)
    report_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "qa_siamese_bilstm_model.keras")
    save_tokenizer(tokenizer, model_dir / "tokenizer.json")

    metadata = {
        "max_vocabulary_size": max_vocabulary_size,
        "effective_vocabulary_size": effective_vocab_size,
        "max_sequence_length": max_sequence_length,
        "embedding_dimension": embedding_dimension,
        "bilstm_units_per_direction": bilstm_units,
        "prediction_threshold": threshold,
        "label_mapping": {"0": "No Match", "1": "Match"},
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "class_weights": class_weights,
        "evaluation_metrics": metrics,
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(report_dir / "training_history.csv", index=False)
    pd.DataFrame(
        {
            "text_a": test_df["text_a"].values,
            "text_b": test_df["text_b"].values,
            "true_label": test_y,
            "match_probability": test_probability,
            "predicted_label": (test_probability >= threshold).astype(int),
        }
    ).to_csv(report_dir / "prediction_analysis.csv", index=False)
    (report_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metadata
