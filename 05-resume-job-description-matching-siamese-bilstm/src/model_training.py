from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, ProjectConfig
from .model_evaluation import baseline_comparison, binary_metrics, tune_threshold
from .sequence_generation import prepare_pair_inputs
from .siamese_model import build_siamese_bilstm
from .tokenizer_utils import fit_shared_tokenizer, save_tokenizer, tokenizer_metadata, vocabulary_size
from .visualization import save_dataset_figures, save_evaluation_figures, save_training_curves


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        try:
            import keras
            keras.utils.set_random_seed(seed)
        except Exception:
            pass


def train_from_pairs(pairs: pd.DataFrame, *, config: ProjectConfig = CONFIG) -> dict[str, Any]:
    required = {"resume_text", "job_description", "label", "split"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Training pairs must contain {sorted(required)}")

    set_reproducibility(config.random_seed)
    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    validation = pairs[pairs["split"] == "validation"].reset_index(drop=True)
    test = pairs[pairs["split"] == "test"].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Train, validation, and test partitions must all be non-empty.")

    tokenizer = fit_shared_tokenizer(
        train["resume_text"].tolist() + train["job_description"].tolist(),
        num_words=config.max_vocabulary_size,
    )
    vocab_size = vocabulary_size(tokenizer, config.max_vocabulary_size)

    def inputs(frame: pd.DataFrame):
        return prepare_pair_inputs(
            tokenizer,
            frame["resume_text"].tolist(),
            frame["job_description"].tolist(),
            max_length=config.max_sequence_length,
        )

    train_resume, train_job = inputs(train)
    val_resume, val_job = inputs(validation)
    test_resume, test_job = inputs(test)

    model = build_siamese_bilstm(
        vocabulary_size=vocab_size,
        max_length=config.max_sequence_length,
        embedding_dimension=config.embedding_dimension,
        bilstm_units=config.bilstm_units,
        projection_dimension=config.projection_dimension,
        dropout_rate=config.dropout_rate,
        learning_rate=config.learning_rate,
    )

    try:
        from tensorflow import keras
    except ImportError:
        import keras  # type: ignore

    config.models_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        [train_resume, train_job],
        train["label"].to_numpy(dtype="float32"),
        validation_data=([val_resume, val_job], validation["label"].to_numpy(dtype="float32")),
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    val_probabilities = model.predict([val_resume, val_job], verbose=0).reshape(-1)
    threshold_result = tune_threshold(validation["label"], val_probabilities)
    test_probabilities = model.predict([test_resume, test_job], verbose=0).reshape(-1)
    metrics = binary_metrics(test["label"], test_probabilities, threshold=threshold_result.threshold)

    model.save(config.model_path)
    save_tokenizer(tokenizer, config.tokenizer_path)

    prediction_frame = test.copy()
    prediction_frame["match_probability"] = test_probabilities
    prediction_frame["predicted_label"] = (test_probabilities >= threshold_result.threshold).astype(int)
    prediction_frame["is_correct"] = (prediction_frame["predicted_label"] == prediction_frame["label"]).astype(int)

    baseline_frame = baseline_comparison(validation, test)
    neural_row = {
        "model": "Shared Siamese BiLSTM",
        **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
    }
    comparison = pd.concat([baseline_frame, pd.DataFrame([neural_row])], ignore_index=True)

    metadata = {
        "project_name": "05-resume-job-description-matching-siamese-bilstm",
        "model_type": "Shared-weight Siamese Bidirectional LSTM",
        "architecture": {
            "shared_encoder": True,
            "embedding_dimension": config.embedding_dimension,
            "bilstm_units": config.bilstm_units,
            "projection_dimension": config.projection_dimension,
            "comparison_features": ["resume vector", "job vector", "absolute difference", "element-wise product", "cosine similarity"],
        },
        "tokenization": {
            **tokenizer_metadata(tokenizer, maximum=config.max_vocabulary_size),
            "max_sequence_length": config.max_sequence_length,
            "shared_tokenizer": True,
            "padding": "post",
            "truncation": "post",
        },
        "prediction_threshold": threshold_result.threshold,
        "label_mapping": {"0": "No Match", "1": "Match"},
        "score_bands": {
            "weak": [0.0, 0.39],
            "moderate": [0.40, 0.69],
            "strong": [0.70, 1.0],
        },
        "training": {
            "seed": config.random_seed,
            "epochs_requested": config.epochs,
            "epochs_completed": len(history.history.get("loss", [])),
            "batch_size": config.batch_size,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "data_note": "Small synthetic demonstration pairs derived from the supplied eight-row example resume dataset. Results are not production evidence.",
        },
        "test_metrics": metrics,
        "responsible_use": "Educational portfolio demonstration only. Not a hiring decision system.",
        "blend_weights": {"neural": 0.35, "tfidf": 0.35, "skill_overlap": 0.30},
    }
    config.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    output_metrics = config.outputs_dir / "metrics"
    output_predictions = config.outputs_dir / "predictions"
    output_figures = config.outputs_dir / "figures"
    for directory in [output_metrics, output_predictions, output_figures]:
        directory.mkdir(parents=True, exist_ok=True)

    (output_metrics / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    comparison.to_csv(output_metrics / "baseline_comparison.csv", index=False)
    pd.DataFrame(history.history).to_csv(output_metrics / "training_history.csv", index=False)
    prediction_frame.to_csv(output_predictions / "sample_predictions.csv", index=False)

    save_training_curves(history.history, output_figures)
    save_evaluation_figures(test["label"], test_probabilities, threshold=threshold_result.threshold, output_dir=output_figures)
    save_dataset_figures(pairs, output_figures)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metadata": metadata,
        "metrics": metrics,
        "history": history.history,
        "predictions": prediction_frame,
        "comparison": comparison,
    }
