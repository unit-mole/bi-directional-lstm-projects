from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MedicalNLP")
class AttentionLayer(tf.keras.layers.Layer):
    """Trainable temporal attention compatible with the supplied Keras model.

    The layer learns a scalar score for every sequence step, converts the
    scores to normalized weights, and returns a weighted sum of BiLSTM states.
    """

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attention_weight",
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
            name="attention_bias",
        )
        super().build(input_shape)

    def compute_attention_weights(self, inputs, mask=None):
        score = tf.tanh(tf.matmul(inputs, self.W) + self.b)

        if mask is not None:
            expanded_mask = tf.cast(tf.expand_dims(mask, axis=-1), score.dtype)
            score = score + (1.0 - expanded_mask) * tf.constant(-1e9, score.dtype)

        return tf.nn.softmax(score, axis=1)

    def call(self, inputs, mask=None):
        attention_weights = self.compute_attention_weights(inputs, mask=mask)
        context_vector = attention_weights * inputs
        return tf.reduce_sum(context_vector, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super().get_config()
