from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from src.attention_layer import (
    BahdanauAttention,
    masked_sparse_categorical_crossentropy,
    masked_token_accuracy,
)
from src.config import ModelConfig


def build_attention_seq2seq_model(
    *,
    code_vocab_size: int,
    comment_vocab_size: int,
    config: ModelConfig,
) -> keras.Model:
    encoder_inputs = keras.Input((config.max_code_len,), dtype="int32", name="encoder_inputs")
    encoder_embedding = keras.layers.Embedding(
        code_vocab_size,
        config.embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )(encoder_inputs)
    encoder_bilstm = keras.layers.Bidirectional(
        keras.layers.LSTM(
            config.encoder_units,
            return_sequences=True,
            return_state=True,
            dropout=config.dropout,
            name="encoder_lstm",
        ),
        name="bidirectional_encoder",
    )
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_embedding)
    state_h = keras.layers.Concatenate(name="encoder_state_h")([forward_h, backward_h])
    state_c = keras.layers.Concatenate(name="encoder_state_c")([forward_c, backward_c])

    decoder_inputs = keras.Input((config.max_comment_len - 1,), dtype="int32", name="decoder_inputs")
    decoder_embedding_layer = keras.layers.Embedding(
        comment_vocab_size,
        config.embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm_layer = keras.layers.LSTM(
        config.decoder_units,
        return_sequences=True,
        return_state=True,
        dropout=config.dropout,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm_layer(decoder_embedding, initial_state=[state_h, state_c])

    attention_layer = BahdanauAttention(config.decoder_units, name="bahdanau_attention")
    context = attention_layer([decoder_outputs, encoder_outputs])
    fused = keras.layers.Concatenate(name="attention_concat")([decoder_outputs, context])
    fused = keras.layers.Dropout(config.dropout, name="decoder_dropout")(fused)
    logits = keras.layers.Dense(comment_vocab_size, activation="softmax", name="token_classifier")(fused)

    return keras.Model([encoder_inputs, decoder_inputs], logits, name="code_comment_bilstm_attention")


def compile_model(model: keras.Model, *, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy,
        metrics=[masked_token_accuracy],
    )
    return model


def build_attention_inference_models(training_model: keras.Model, config: ModelConfig):
    encoder_outputs = training_model.get_layer("bidirectional_encoder").output[0]
    state_h = training_model.get_layer("encoder_state_h").output
    state_c = training_model.get_layer("encoder_state_c").output
    encoder_model = keras.Model(training_model.inputs[0], [encoder_outputs, state_h, state_c], name="encoder_inference")

    token_input = keras.Input((1,), dtype="int32", name="decoder_token_input")
    encoder_output_input = keras.Input(
        (config.max_code_len, config.decoder_units), dtype="float32", name="encoder_output_input"
    )
    state_h_input = keras.Input((config.decoder_units,), name="state_h_input")
    state_c_input = keras.Input((config.decoder_units,), name="state_c_input")

    embedding_layer = training_model.get_layer("decoder_embedding")
    decoder_lstm = training_model.get_layer("decoder_lstm")
    attention = training_model.get_layer("bahdanau_attention")
    concat = training_model.get_layer("attention_concat")
    dropout = training_model.get_layer("decoder_dropout")
    classifier = training_model.get_layer("token_classifier")

    embedded = embedding_layer(token_input)
    decoder_output, next_h, next_c = decoder_lstm(
        embedded, initial_state=[state_h_input, state_c_input], training=False
    )
    context, scores = attention(
        [decoder_output, encoder_output_input], return_attention_scores=True
    )
    fused = concat([decoder_output, context])
    fused = dropout(fused, training=False)
    probabilities = classifier(fused)
    decoder_model = keras.Model(
        [token_input, encoder_output_input, state_h_input, state_c_input],
        [probabilities, next_h, next_c, scores],
        name="decoder_inference",
    )
    return encoder_model, decoder_model


def load_attention_model(path: str | Path) -> keras.Model:
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "BahdanauAttention": BahdanauAttention,
            "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
            "masked_token_accuracy": masked_token_accuracy,
        },
    )


def build_legacy_inference_models(full_model: keras.Model):
    """Reconstruct one-step inference models from the supplied non-attention checkpoint."""
    encoder_model = keras.Model(
        full_model.inputs[0],
        [
            full_model.get_layer("encoder_state_h").output,
            full_model.get_layer("encoder_state_c").output,
        ],
        name="legacy_encoder_inference",
    )
    units = int(full_model.get_layer("decoder_lstm").units)
    token_input = keras.Input((1,), dtype="int32", name="decoder_token_input")
    state_h_input = keras.Input((units,), name="legacy_state_h")
    state_c_input = keras.Input((units,), name="legacy_state_c")
    embedded = full_model.get_layer("decoder_embedding")(token_input)
    output, next_h, next_c = full_model.get_layer("decoder_lstm")(
        embedded, initial_state=[state_h_input, state_c_input], training=False
    )
    probabilities = full_model.get_layer("decoder_output")(output)
    decoder_model = keras.Model(
        [token_input, state_h_input, state_c_input],
        [probabilities, next_h, next_c],
        name="legacy_decoder_inference",
    )
    return encoder_model, decoder_model
