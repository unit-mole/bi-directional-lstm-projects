from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # Enables local artifact generation with Keras' torch backend.
    import keras  # type: ignore


@keras.saving.register_keras_serializable(package="ResumeJD")
class AbsoluteDifference(keras.layers.Layer):
    """Serializable |a-b| comparison layer."""

    def call(self, inputs: list[Any]) -> Any:
        left, right = inputs
        return keras.ops.abs(left - right)

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


def build_shared_encoder(
    *,
    vocabulary_size: int,
    max_length: int,
    embedding_dimension: int,
    bilstm_units: int,
    projection_dimension: int,
    dropout_rate: float,
) -> "keras.Model":
    encoder_input = keras.Input(shape=(max_length,), dtype="int32", name="shared_text_input")
    x = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dimension,
        mask_zero=False,
        name="shared_embedding",
    )(encoder_input)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(bilstm_units, return_sequences=True),
        name="shared_bilstm",
    )(x)
    max_pool = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    avg_pool = keras.layers.GlobalAveragePooling1D(name="global_average_pool")(x)
    x = keras.layers.Concatenate(name="pooled_context")([max_pool, avg_pool])
    x = keras.layers.Dropout(dropout_rate, name="encoder_dropout")(x)
    x = keras.layers.Dense(projection_dimension, activation="relu", name="semantic_projection")(x)
    x = keras.layers.UnitNormalization(axis=1, name="normalized_embedding")(x)
    return keras.Model(encoder_input, x, name="shared_bilstm_encoder")


def build_siamese_bilstm(
    *,
    vocabulary_size: int,
    max_length: int,
    embedding_dimension: int = 32,
    bilstm_units: int = 12,
    projection_dimension: int = 32,
    dropout_rate: float = 0.25,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    resume_input = keras.Input(shape=(max_length,), dtype="int32", name="resume_input")
    job_input = keras.Input(shape=(max_length,), dtype="int32", name="job_description_input")

    shared_encoder = build_shared_encoder(
        vocabulary_size=vocabulary_size,
        max_length=max_length,
        embedding_dimension=embedding_dimension,
        bilstm_units=bilstm_units,
        projection_dimension=projection_dimension,
        dropout_rate=dropout_rate,
    )
    resume_vector = shared_encoder(resume_input)
    job_vector = shared_encoder(job_input)

    difference = AbsoluteDifference(name="absolute_difference")([resume_vector, job_vector])
    product = keras.layers.Multiply(name="elementwise_product")([resume_vector, job_vector])
    cosine = keras.layers.Dot(axes=1, normalize=True, name="cosine_similarity")([resume_vector, job_vector])

    merged = keras.layers.Concatenate(name="comparison_features")(
        [resume_vector, job_vector, difference, product, cosine]
    )
    x = keras.layers.Dense(96, activation="relu", name="matching_dense_1")(merged)
    x = keras.layers.Dropout(dropout_rate, name="matching_dropout_1")(x)
    x = keras.layers.Dense(48, activation="relu", name="matching_dense_2")(x)
    x = keras.layers.Dropout(dropout_rate / 2, name="matching_dropout_2")(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="match_probability")(x)

    model = keras.Model([resume_input, job_input], output, name="resume_job_siamese_bilstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_siamese_model(path: str | Path, *, compile_model: bool = False):
    return keras.models.load_model(
        path,
        custom_objects={"AbsoluteDifference": AbsoluteDifference},
        compile=compile_model,
        safe_mode=True,
    )
