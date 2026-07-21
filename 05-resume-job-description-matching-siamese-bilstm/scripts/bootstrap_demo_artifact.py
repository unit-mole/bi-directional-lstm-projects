"""Create the included demonstration Keras artifact without TensorFlow.

This script exists only so the portfolio package can ship with a small, runnable
model artifact in environments where TensorFlow is unavailable. It trains the
same mathematical architecture in PyTorch, transfers the learned arrays into
the identical Keras model, verifies prediction parity, and saves a backend-neutral
`.keras` file. Normal project users should prefer `scripts/train_model.py`, which
trains directly through TensorFlow/Keras.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# Avoid very slow oneDNN thread behavior in constrained artifact-generation environments.
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.config import CONFIG
from src.model_evaluation import baseline_comparison, binary_metrics, ranking_metrics, tune_threshold
from src.pair_generation import generate_balanced_pairs
from src.sequence_generation import prepare_pair_inputs
from src.tokenizer_utils import fit_shared_tokenizer, save_tokenizer, tokenizer_metadata, vocabulary_size
from src.visualization import save_architecture_diagram, save_dataset_figures, save_evaluation_figures, save_training_curves


class TorchSiameseBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, units: int, projection_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, units, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(units * 4, projection_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(projection_dim * 4 + 1, 96)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(96, 48)
        self.dropout3 = nn.Dropout(dropout / 2)
        self.output = nn.Linear(48, 1)

    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.embedding(sequence)
        x, _ = self.bilstm(x)
        max_pool = torch.amax(x, dim=1)
        average_pool = torch.mean(x, dim=1)
        x = torch.cat([max_pool, average_pool], dim=1)
        x = self.dropout1(x)
        x = F.relu(self.projection(x))
        return F.normalize(x, p=2, dim=1)

    def forward(self, resume: torch.Tensor, job: torch.Tensor) -> torch.Tensor:
        resume_vector = self.encode(resume)
        job_vector = self.encode(job)
        difference = torch.abs(resume_vector - job_vector)
        product = resume_vector * job_vector
        cosine = F.cosine_similarity(resume_vector, job_vector, dim=1).unsqueeze(1)
        merged = torch.cat([resume_vector, job_vector, difference, product, cosine], dim=1)
        x = F.relu(self.dense1(merged))
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        x = self.dropout3(x)
        return self.output(x).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def arrays(frame: pd.DataFrame, tokenizer):
    return prepare_pair_inputs(
        tokenizer,
        frame["resume_text"].tolist(),
        frame["job_description"].tolist(),
        max_length=CONFIG.max_sequence_length,
    )


def predict(model: nn.Module, resume: np.ndarray, job: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(resume).long(), torch.from_numpy(job).long())
        return torch.sigmoid(logits).cpu().numpy()


def transfer_to_keras(torch_model: TorchSiameseBiLSTM, vocab_size: int):
    # The environment invoking this script must set KERAS_BACKEND=torch.
    from src.siamese_model import build_siamese_bilstm

    keras_model = build_siamese_bilstm(
        vocabulary_size=vocab_size,
        max_length=CONFIG.max_sequence_length,
        embedding_dimension=CONFIG.embedding_dimension,
        bilstm_units=CONFIG.bilstm_units,
        projection_dimension=CONFIG.projection_dimension,
        dropout_rate=CONFIG.dropout_rate,
        learning_rate=CONFIG.learning_rate,
    )

    encoder = keras_model.get_layer("shared_bilstm_encoder")
    encoder.get_layer("shared_embedding").set_weights([
        torch_model.embedding.weight.detach().cpu().numpy()
    ])

    state = torch_model.bilstm.state_dict()
    forward_weights = [
        state["weight_ih_l0"].cpu().numpy().T,
        state["weight_hh_l0"].cpu().numpy().T,
        (state["bias_ih_l0"] + state["bias_hh_l0"]).cpu().numpy(),
    ]
    backward_weights = [
        state["weight_ih_l0_reverse"].cpu().numpy().T,
        state["weight_hh_l0_reverse"].cpu().numpy().T,
        (state["bias_ih_l0_reverse"] + state["bias_hh_l0_reverse"]).cpu().numpy(),
    ]
    encoder.get_layer("shared_bilstm").set_weights(forward_weights + backward_weights)
    encoder.get_layer("semantic_projection").set_weights([
        torch_model.projection.weight.detach().cpu().numpy().T,
        torch_model.projection.bias.detach().cpu().numpy(),
    ])
    keras_model.get_layer("matching_dense_1").set_weights([
        torch_model.dense1.weight.detach().cpu().numpy().T,
        torch_model.dense1.bias.detach().cpu().numpy(),
    ])
    keras_model.get_layer("matching_dense_2").set_weights([
        torch_model.dense2.weight.detach().cpu().numpy().T,
        torch_model.dense2.bias.detach().cpu().numpy(),
    ])
    keras_model.get_layer("match_probability").set_weights([
        torch_model.output.weight.detach().cpu().numpy().T,
        torch_model.output.bias.detach().cpu().numpy(),
    ])
    return keras_model


def main() -> None:
    set_seed(CONFIG.random_seed)
    print("[1] seed", flush=True)
    resumes = pd.read_csv(CONFIG.data_dir / "sample" / "sample_resumes.csv")
    jobs = pd.read_csv(CONFIG.data_dir / "sample" / "sample_job_descriptions.csv")
    pairs = generate_balanced_pairs(resumes, jobs, seed=CONFIG.random_seed)
    print("[2] pairs", pairs.shape, flush=True)
    CONFIG.training_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(CONFIG.training_pairs_path, index=False)

    train = pairs[pairs["split"] == "train"].reset_index(drop=True)
    validation = pairs[pairs["split"] == "validation"].reset_index(drop=True)
    test = pairs[pairs["split"] == "test"].reset_index(drop=True)

    tokenizer = fit_shared_tokenizer(
        train["resume_text"].tolist() + train["job_description"].tolist(),
        num_words=CONFIG.max_vocabulary_size,
    )
    vocab_size = vocabulary_size(tokenizer, CONFIG.max_vocabulary_size)
    print("[3] tokenizer", vocab_size, flush=True)
    train_r, train_j = arrays(train, tokenizer)
    val_r, val_j = arrays(validation, tokenizer)
    test_r, test_j = arrays(test, tokenizer)

    print("[4] arrays", train_r.shape, val_r.shape, test_r.shape, flush=True)
    model = TorchSiameseBiLSTM(
        vocab_size,
        CONFIG.embedding_dimension,
        CONFIG.bilstm_units,
        CONFIG.projection_dimension,
        CONFIG.dropout_rate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    x_train_r = torch.from_numpy(train_r).long()
    x_train_j = torch.from_numpy(train_j).long()
    y_train = torch.from_numpy(train["label"].to_numpy(dtype="float32"))
    x_val_r = torch.from_numpy(val_r).long()
    x_val_j = torch.from_numpy(val_j).long()
    y_val = torch.from_numpy(validation["label"].to_numpy(dtype="float32"))

    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    best_state = None
    best_val = float("inf")
    best_val_accuracy = -1.0
    stale = 0
    for epoch in range(1, 81):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_r, x_train_j)
        loss = criterion(logits, y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_prob = torch.sigmoid(model(x_train_r, x_train_j))
            val_logits = model(x_val_r, x_val_j)
            val_loss = criterion(val_logits, y_val)
            val_prob = torch.sigmoid(val_logits)
            train_acc = ((train_prob >= 0.5) == (y_train >= 0.5)).float().mean()
            val_acc = ((val_prob >= 0.5) == (y_val >= 0.5)).float().mean()

        history["loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss.item()))
        history["accuracy"].append(float(train_acc.item()))
        history["val_accuracy"].append(float(val_acc.item()))

        current_val_accuracy = float(val_acc.item())
        current_val_loss = float(val_loss.item())
        improved = (
            current_val_accuracy > best_val_accuracy + 1e-8
            or (abs(current_val_accuracy - best_val_accuracy) <= 1e-8 and current_val_loss < best_val)
        )
        if improved:
            best_val_accuracy = current_val_accuracy
            best_val = current_val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 20 and epoch >= 40:
            break

    print("[5] trained", len(history["loss"]), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    val_probabilities = predict(model, val_r, val_j)
    tuned = tune_threshold(validation["label"], val_probabilities)
    test_probabilities = predict(model, test_r, test_j)
    metrics = binary_metrics(test["label"], test_probabilities, threshold=tuned.threshold)
    print("[6] metrics", metrics, flush=True)

    keras_model = transfer_to_keras(model, vocab_size)
    print("[7] converted", flush=True)
    keras_output = keras_model([test_r, test_j], training=False)
    if hasattr(keras_output, "detach"):
        keras_probabilities = keras_output.detach().cpu().numpy().reshape(-1)
    else:
        keras_probabilities = np.asarray(keras_output).reshape(-1)
    max_difference = float(np.max(np.abs(keras_probabilities - test_probabilities)))
    print("[8] parity", max_difference, flush=True)
    if max_difference > 1e-4:
        raise RuntimeError(f"Keras transfer parity check failed: max difference {max_difference}")

    CONFIG.models_dir.mkdir(parents=True, exist_ok=True)
    keras_model.save(CONFIG.model_path)
    save_tokenizer(tokenizer, CONFIG.tokenizer_path)
    print("[9] saved artifacts", flush=True)

    prediction_frame = test.copy()
    prediction_frame["match_probability"] = keras_probabilities
    prediction_frame["predicted_label"] = (keras_probabilities >= tuned.threshold).astype(int)
    prediction_frame["is_correct"] = (prediction_frame["predicted_label"] == prediction_frame["label"]).astype(int)

    baseline_frame = baseline_comparison(validation, test)
    comparison = pd.concat([
        baseline_frame,
        pd.DataFrame([{
            "model": "Shared Siamese BiLSTM",
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        }]),
    ], ignore_index=True)

    # Category-level ranking evaluation on held-out test jobs.
    ranking_rows = []
    for _, job_row in jobs[jobs["split"] == "test"].iterrows():
        repeated_job = [job_row["job_description"]] * len(resumes)
        rr, jj = prepare_pair_inputs(
            tokenizer,
            resumes["resume_text"].tolist(),
            repeated_job,
            max_length=CONFIG.max_sequence_length,
        )
        ranking_output = keras_model([rr, jj], training=False)
        if hasattr(ranking_output, "detach"):
            scores = ranking_output.detach().cpu().numpy().reshape(-1)
        else:
            scores = np.asarray(ranking_output).reshape(-1)
        for resume_row, score in zip(resumes.to_dict(orient="records"), scores):
            ranking_rows.append({
                "job_id": job_row["job_id"],
                "job_category": job_row["category"],
                "resume_id": resume_row["resume_id"],
                "resume_category": resume_row["category"],
                "score": float(score),
            })
    ranking_frame = pd.DataFrame(ranking_rows)
    rank_metrics = ranking_metrics(ranking_frame, top_k=3)
    print("[10] ranking", rank_metrics, flush=True)

    metadata = {
        "project_name": "05-resume-job-description-matching-siamese-bilstm",
        "model_type": "Shared-weight Siamese Bidirectional LSTM",
        "artifact_format": "Keras v3 .keras",
        "architecture": {
            "shared_encoder": True,
            "embedding_dimension": CONFIG.embedding_dimension,
            "bilstm_units": CONFIG.bilstm_units,
            "projection_dimension": CONFIG.projection_dimension,
            "comparison_features": [
                "resume vector", "job vector", "absolute difference",
                "element-wise product", "cosine similarity"
            ],
        },
        "tokenization": {
            **tokenizer_metadata(tokenizer, maximum=CONFIG.max_vocabulary_size),
            "max_sequence_length": CONFIG.max_sequence_length,
            "shared_tokenizer": True,
            "padding": "post",
            "truncation": "post",
        },
        "prediction_threshold": tuned.threshold,
        "label_mapping": {"0": "No Match", "1": "Match"},
        "score_bands": {"weak": [0.0, 0.39], "moderate": [0.40, 0.69], "strong": [0.70, 1.0]},
        "training": {
            "seed": CONFIG.random_seed,
            "epochs_completed": len(history["loss"]),
            "batch_style": "full-batch demonstration training",
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "artifact_bootstrap_note": (
                "The included backend-neutral Keras artifact was initialized with arrays trained in the "
                "mathematically equivalent PyTorch implementation because TensorFlow was unavailable in "
                "the artifact-generation environment. scripts/train_model.py remains the standard "
                "TensorFlow/Keras training path."
            ),
            "data_note": (
                "Small synthetic demonstration pairs derived from the supplied eight-row example resume "
                "dataset. Results are not production evidence."
            ),
        },
        "test_metrics": metrics,
        "ranking_metrics": rank_metrics,
        "keras_transfer_max_probability_difference": max_difference,
        "responsible_use": "Educational portfolio demonstration only. Not a hiring decision system.",
        "blend_weights": {"neural": 0.35, "tfidf": 0.35, "skill_overlap": 0.30},
    }
    CONFIG.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_dir = CONFIG.outputs_dir / "metrics"
    predictions_dir = CONFIG.outputs_dir / "predictions"
    figures_dir = CONFIG.outputs_dir / "figures"
    for directory in [metrics_dir, predictions_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (metrics_dir / "ranking_metrics.json").write_text(json.dumps(rank_metrics, indent=2), encoding="utf-8")
    comparison.to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(history).to_csv(metrics_dir / "training_history.csv", index=False)
    prediction_frame.to_csv(predictions_dir / "sample_predictions.csv", index=False)
    ranking_frame.sort_values(["job_id", "score"], ascending=[True, False]).to_csv(
        predictions_dir / "ranking_examples.csv", index=False
    )

    print("[11] figure generation deferred to scripts/generate_figures.py", flush=True)

    print(json.dumps({"metrics": metrics, "ranking": rank_metrics, "epochs": len(history["loss"])}, indent=2))
    print(f"Saved model to {CONFIG.model_path}")


if __name__ == "__main__":
    main()
