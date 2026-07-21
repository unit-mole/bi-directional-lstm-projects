"""Build and train a true BiLSTM-with-attention emotion classifier."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, callbacks, layers

from .attention_layer import TemporalAttention
from .config import MODEL_DIR, OUTPUT_DIR, TrainingConfig
from .data_preprocessing import (
    load_and_clean_dataset,
    split_dataframe,
    validate_class_support,
)
from .model_evaluation import evaluate_and_save
from .sequence_generation import texts_to_padded_sequences
from .tokenizer_utils import create_tokenizer, effective_vocabulary_size, save_tokenizer
from .visualization import save_eda_outputs, save_training_curves


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_bilstm_attention_model(
    vocabulary_size: int,
    number_of_classes: int,
    config: TrainingConfig,
) -> Model:
    """Create the embedding → BiLSTM → attention → softmax architecture."""

    inputs = layers.Input(
        shape=(config.max_sequence_length,),
        dtype="int32",
        name="emotion_text_input",
    )
    embeddings = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=config.embedding_dimension,
        mask_zero=True,
        name="embedding",
    )(inputs)
    embeddings = layers.SpatialDropout1D(config.dropout_rate, name="embedding_dropout")(embeddings)
    sequence_features = layers.Bidirectional(
        layers.LSTM(
            config.lstm_units,
            return_sequences=True,
            dropout=config.dropout_rate,
            recurrent_dropout=config.recurrent_dropout,
        ),
        name="bilstm_encoder",
    )(embeddings)
    context_vector, _attention_scores = TemporalAttention(name="temporal_attention")(sequence_features)
    hidden = layers.Dense(config.dense_units, activation="relu", name="classification_dense")(context_vector)
    hidden = layers.Dropout(config.dropout_rate, name="classification_dropout")(hidden)
    outputs = layers.Dense(number_of_classes, activation="softmax", name="emotion_output")(hidden)

    model = Model(inputs=inputs, outputs=outputs, name="emotion_detection_bilstm_attention")
    top_k = min(3, number_of_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k, name=f"top_{top_k}_accuracy"),
        ],
    )
    return model


def _label_mapping(label_encoder: LabelEncoder) -> dict[str, str]:
    return {str(index): str(label) for index, label in enumerate(label_encoder.classes_)}


def train_pipeline(
    data_path: str | Path,
    config: TrainingConfig | None = None,
    model_dir: str | Path = MODEL_DIR,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Train, evaluate, and save every artifact needed by the Streamlit app."""

    config = config or TrainingConfig()
    set_reproducibility(config.random_seed)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, audit = load_and_clean_dataset(data_path)
    validate_class_support(frame, config.minimum_samples_per_class)
    train_df, validation_df, test_df = split_dataframe(
        frame,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_seed=config.random_seed,
    )

    # Leakage prevention: fit encoder/tokenizer using the training partition only.
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["emotion"])
    unknown_validation = set(validation_df["emotion"]) - set(label_encoder.classes_)
    unknown_test = set(test_df["emotion"]) - set(label_encoder.classes_)
    if unknown_validation or unknown_test:
        raise ValueError("A split contains a class absent from training. Increase dataset support per class.")

    tokenizer = create_tokenizer(train_df["text_clean"], config.max_vocab_size)
    vocabulary_size = effective_vocabulary_size(tokenizer, config.max_vocab_size)

    x_train = texts_to_padded_sequences(train_df["text_clean"], tokenizer, config.max_sequence_length)
    x_validation = texts_to_padded_sequences(
        validation_df["text_clean"], tokenizer, config.max_sequence_length
    )
    x_test = texts_to_padded_sequences(test_df["text_clean"], tokenizer, config.max_sequence_length)
    y_train = label_encoder.transform(train_df["emotion"])
    y_validation = label_encoder.transform(validation_df["emotion"])
    y_test = label_encoder.transform(test_df["emotion"])

    class_values = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=class_values, y=y_train)
    class_weights = {int(class_id): float(weight) for class_id, weight in zip(class_values, weights)}

    model = build_bilstm_attention_model(vocabulary_size, len(label_encoder.classes_), config)
    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_dir / "best_emotion_bilstm_attention_model.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callback_list,
        class_weight=class_weights,
        verbose=1,
    )

    model_path = model_dir / "emotion_bilstm_attention_model.keras"
    model.save(model_path)
    save_tokenizer(tokenizer, model_dir / "tokenizer.json")
    mapping = _label_mapping(label_encoder)
    (model_dir / "label_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    save_training_curves(history_frame, output_dir)
    save_eda_outputs(frame, output_dir)

    metrics = evaluate_and_save(
        model=model,
        x_test=x_test,
        y_test=y_test,
        test_frame=test_df,
        label_mapping=mapping,
        output_dir=output_dir,
    )

    metadata = {
        "project": "01-emotion-detection-bilstm-attention",
        "model_type": "Bidirectional LSTM with Temporal Attention",
        "artifact_status": "trained_attention_model",
        "model_path": model_path.name,
        "tokenizer_path": "tokenizer.json",
        "label_mapping_path": "label_mapping.json",
        "text_column": audit.text_column,
        "target_column": audit.label_column,
        "classes": list(label_encoder.classes_),
        "number_of_classes": len(label_encoder.classes_),
        "vocabulary_size": vocabulary_size,
        "max_sequence_length": config.max_sequence_length,
        "embedding_dimension": config.embedding_dimension,
        "lstm_units": config.lstm_units,
        "class_weights": class_weights,
        "training_config": config.to_dict(),
        "dataset_audit": audit.to_dict(),
        "split_rows": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "evaluation_metrics": metrics,
        "responsible_use": (
            "Educational portfolio model only; predictions are estimates and not suitable "
            "as the sole basis for high-stakes decisions."
        ),
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
