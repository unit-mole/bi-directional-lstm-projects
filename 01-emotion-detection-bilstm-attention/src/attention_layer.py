"""Serializable temporal attention layer for BiLSTM sequence outputs."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="EmotionNLP")
class TemporalAttention(layers.Layer):
    """Learn a normalized importance weight for every sequence timestep.

    The layer returns both the weighted context vector and the attention scores.
    The classifier consumes the context vector while the scores remain available
    for optional token-level visualization.
    """

    def __init__(self, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.supports_masking = True

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.score_kernel = self.add_weight(
            name="score_kernel",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.score_bias = None
        if self.use_bias:
            self.score_bias = self.add_weight(
                name="score_bias",
                shape=(1,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        logits = tf.squeeze(tf.matmul(inputs, self.score_kernel), axis=-1)
        if self.score_bias is not None:
            logits = logits + self.score_bias
        logits = tf.tanh(logits)

        if mask is not None:
            mask = tf.cast(mask, logits.dtype)
            logits = logits + (1.0 - mask) * tf.cast(-1e9, logits.dtype)

        attention_scores = tf.nn.softmax(logits, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(attention_scores, axis=-1), axis=1)
        return context, attention_scores

    def compute_mask(self, inputs, mask=None):
        return (None, None)

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias})
        return config
