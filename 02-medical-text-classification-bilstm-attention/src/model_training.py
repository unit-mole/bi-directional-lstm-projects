from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from .attention_layer import AttentionLayer
from .config import TrainingConfig
from .medical_text_preprocessing import clean_medical_text
from .sequence_generation import pad_integer_sequences
from .tokenizer_utils import build_tokenizer, save_tokenizer


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str))
    return encoded.astype(np.int32), encoder


def _stratify_or_none(labels: np.ndarray):
    counts = pd.Series(labels).value_counts()
    return labels if len(counts) > 1 and int(counts.min()) >= 2 else None


def split_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    test_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[str, np.ndarray]:
    if len(features) != len(targets):
        raise ValueError("features and targets must contain the same row count.")
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("test_fraction and validation_fraction must be in (0, 1).")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be below 1.")

    temp_fraction = test_fraction + validation_fraction
    unique_classes = np.unique(targets)
    minimum_temp_rows = len(unique_classes)
    requested_temp_rows = int(np.ceil(len(targets) * temp_fraction))

    if requested_temp_rows < minimum_temp_rows:
        temp_fraction = minimum_temp_rows / len(targets)

    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        targets,
        test_size=temp_fraction,
        random_state=random_seed,
        stratify=_stratify_or_none(targets),
    )

    relative_test_fraction = test_fraction / (test_fraction + validation_fraction)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_fraction,
        random_state=random_seed,
        stratify=_stratify_or_none(y_temp),
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
    }


def build_bilstm_attention_model(
    *,
    vocabulary_size: int,
    number_of_classes: int,
    config: TrainingConfig,
) -> tf.keras.Model:
    text_input = tf.keras.layers.Input(
        shape=(config.max_sequence_length,),
        dtype="int32",
        name="clinical_text_input",
    )
    x = tf.keras.layers.Embedding(
        vocabulary_size,
        config.embedding_dimension,
        mask_zero=False,
        name="embedding",
    )(text_input)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            config.bilstm_units,
            return_sequences=True,
            dropout=0.10,
        ),
        name="bilstm",
    )(x)
    x = AttentionLayer(name="attention")(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout_1")(x)
    x = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        name="dense_1",
    )(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_2")(x)
    output = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="class_output",
    )(x)

    model = tf.keras.Model(
        inputs=text_input,
        outputs=output,
        name="medical_text_bilstm_attention_classifier",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(3, number_of_classes),
                name="top_k_accuracy",
            ),
        ],
    )
    return model


def compute_class_weights(targets: np.ndarray) -> dict[int, float]:
    classes = np.unique(targets)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=targets,
    )
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def train_model(
    dataframe: pd.DataFrame,
    *,
    model_directory: str | Path,
    output_directory: str | Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    set_reproducibility(config.random_seed)
    model_directory = Path(model_directory)
    output_directory = Path(output_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not {"clinical_text", "label"}.issubset(dataframe.columns):
        raise ValueError("Dataframe must include clinical_text and label columns.")

    cleaned_texts = dataframe["clinical_text"].map(
        lambda value: clean_medical_text(
            value,
            mode=config.preprocessing_mode,
        )
    )
    targets, label_encoder = encode_labels(dataframe["label"])

    tokenizer = build_tokenizer(
        cleaned_texts,
        maximum_vocabulary_size=config.max_vocabulary_size,
    )
    sequences = tokenizer.texts_to_sequences(cleaned_texts.tolist())
    features = pad_integer_sequences(
        sequences,
        max_length=config.max_sequence_length,
        padding="post",
        truncating="post",
    )

    splits = split_dataset(
        features,
        targets,
        test_fraction=config.test_fraction,
        validation_fraction=config.validation_fraction,
        random_seed=config.random_seed,
    )

    vocabulary_size = min(
        config.max_vocabulary_size,
        len(tokenizer.word_index) + 1,
    )
    model = build_bilstm_attention_model(
        vocabulary_size=vocabulary_size,
        number_of_classes=len(label_encoder.classes_),
        config=config,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_directory / "best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        splits["X_train"],
        splits["y_train"],
        validation_data=(
            splits["X_validation"],
            splits["y_validation"],
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=compute_class_weights(splits["y_train"]),
        callbacks=callbacks,
        verbose=2,
    )

    model_path = model_directory / "medical_text_bilstm_attention_model.keras"
    model.save(model_path)
    save_tokenizer(tokenizer, model_directory / "tokenizer_config.json")

    label_mapping = {
        str(index): label
        for index, label in enumerate(label_encoder.classes_.tolist())
    }
    (model_directory / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2),
        encoding="utf-8",
    )

    import pickle

    with (model_directory / "label_encoder.pkl").open("wb") as handle:
        pickle.dump(label_encoder, handle)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(
        output_directory / "medical_text_training_history.csv",
        index=False,
    )

    metadata = {
        "project_name": "02-medical-text-classification-bilstm-attention",
        "model_type": "Bidirectional LSTM with temporal attention",
        "class_labels": label_encoder.classes_.tolist(),
        "number_of_classes": len(label_encoder.classes_),
        "max_vocabulary_size": config.max_vocabulary_size,
        "effective_vocabulary_size": vocabulary_size,
        "max_sequence_length": config.max_sequence_length,
        "embedding_dimension": config.embedding_dimension,
        "bilstm_units": config.bilstm_units,
        "preprocessing_mode": config.preprocessing_mode,
        "text_column": "clinical_text",
        "target_column": "label",
        "training_configuration": config.to_dict(),
        "split_sizes": {
            key: int(len(value))
            for key, value in splits.items()
            if key.startswith("y_")
        },
        "artifact_status": "trained_from_local_dataset",
        "medical_disclaimer": (
            "Educational portfolio model only; not a medical diagnostic tool."
        ),
    }
    (model_directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "history": history_frame,
        "splits": splits,
        "metadata": metadata,
    }
