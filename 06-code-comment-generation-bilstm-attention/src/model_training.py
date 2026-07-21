from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tensorflow import keras

from src.config import ModelConfig, ProjectPaths
from src.seq2seq_model import (
    build_attention_inference_models,
    build_attention_seq2seq_model,
    compile_model,
    load_attention_model,
)
from src.sequence_generation import prepare_seq2seq_arrays
from src.tokenizer_utils import build_tokenizers, effective_vocab_size, save_tokenizer


def train_attention_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    config: ModelConfig,
    paths: ProjectPaths,
):
    code_tokenizer, comment_tokenizer = build_tokenizers(
        train_df["code_clean"].tolist(),
        train_df["comment_sequence"].tolist(),
        max_code_vocab=config.max_code_vocab,
        max_comment_vocab=config.max_comment_vocab,
        oov_token=config.oov_token,
    )
    train = prepare_seq2seq_arrays(
        train_df["code_clean"].tolist(), train_df["comment_sequence"].tolist(),
        code_tokenizer, comment_tokenizer,
        max_code_len=config.max_code_len, max_comment_len=config.max_comment_len,
    )
    validation = prepare_seq2seq_arrays(
        validation_df["code_clean"].tolist(), validation_df["comment_sequence"].tolist(),
        code_tokenizer, comment_tokenizer,
        max_code_len=config.max_code_len, max_comment_len=config.max_comment_len,
    )
    code_vocab = effective_vocab_size(code_tokenizer, config.max_code_vocab)
    comment_vocab = effective_vocab_size(comment_tokenizer, config.max_comment_vocab)
    model = build_attention_seq2seq_model(
        code_vocab_size=code_vocab, comment_vocab_size=comment_vocab, config=config
    )
    compile_model(model, learning_rate=config.learning_rate)

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(
            paths.attention_model_path, monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        [train.encoder_inputs, train.decoder_inputs],
        train.decoder_targets,
        validation_data=(
            [validation.encoder_inputs, validation.decoder_inputs],
            validation.decoder_targets,
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    best_model = load_attention_model(paths.attention_model_path)
    encoder_model, decoder_model = build_attention_inference_models(best_model, config)
    encoder_model.save(paths.encoder_model_path)
    decoder_model.save(paths.decoder_model_path)
    save_tokenizer(code_tokenizer, paths.code_tokenizer_path)
    save_tokenizer(comment_tokenizer, paths.comment_tokenizer_path)

    metadata: dict[str, Any] = {
        "project": "06-code-comment-generation-bilstm-attention",
        "checkpoint_kind": "true_bilstm_bahdanau_attention",
        "attention_available": True,
        "language": config.language,
        "preprocessing_mode": config.preprocessing_mode,
        "max_code_len": config.max_code_len,
        "max_comment_len": config.max_comment_len,
        "code_vocab_size": code_vocab,
        "comment_vocab_size": comment_vocab,
        "start_token": config.start_token,
        "end_token": config.end_token,
        "model_config": config.to_dict(),
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
    }
    paths.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(paths.outputs_dir / "training_history_attention.csv", index=False)
    return best_model, history_df, metadata
