from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="CodeCommentGeneration")
class BahdanauAttention(keras.layers.Layer):
    """Additive attention over all encoder time steps."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_projection = keras.layers.Dense(units, use_bias=False, name="query_projection")
        self.value_projection = keras.layers.Dense(units, use_bias=False, name="value_projection")
        self.score_projection = keras.layers.Dense(1, use_bias=False, name="score_projection")

    def call(self, inputs, mask=None, return_attention_scores: bool = False):
        query, values = inputs  # [B, T_dec, D], [B, T_enc, D]
        query_features = self.query_projection(query)[:, :, tf.newaxis, :]
        value_features = self.value_projection(values)[:, tf.newaxis, :, :]
        scores = self.score_projection(tf.nn.tanh(query_features + value_features))
        scores = tf.squeeze(scores, axis=-1)  # [B, T_dec, T_enc]

        value_mask = None
        if isinstance(mask, (list, tuple)) and len(mask) > 1:
            value_mask = mask[1]
        if value_mask is not None:
            scores += (1.0 - tf.cast(value_mask[:, tf.newaxis, :], scores.dtype)) * -1e9

        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(weights, values)
        if return_attention_scores:
            return context, weights
        return context

    def get_config(self):
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="CodeCommentGeneration")
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    per_token = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), per_token.dtype)
    return tf.math.divide_no_nan(tf.reduce_sum(per_token * mask), tf.reduce_sum(mask))


@keras.saving.register_keras_serializable(package="CodeCommentGeneration")
def masked_token_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.cast(tf.equal(y_true, predicted), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(matches * mask), tf.reduce_sum(mask))
