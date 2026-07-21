"""Pure TensorFlow linear-chain CRF operations and a BiLSTM-CRF model.

TensorFlow Addons is deliberately not used because the project has ended active
development. The model saves architecture metadata plus weights and is rebuilt
for inference, avoiding fragile custom-object deserialization.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # Allows lightweight preprocessing/tests before ML dependencies are installed.
    tf = None  # type: ignore[assignment]


def _require_tensorflow():
    if tf is None:
        raise ImportError("TensorFlow is required for CRF training and model inference. Install requirements.txt.")
    return tf


def crf_sequence_score(
    emissions: tf.Tensor,
    tags: tf.Tensor,
    lengths: tf.Tensor,
    transitions: tf.Tensor,
) -> "tf.Tensor":
    _tf = _require_tensorflow()
    emissions = _tf.convert_to_tensor(emissions)
    tags = _tf.cast(tags, _tf.int32)
    lengths = _tf.cast(lengths, _tf.int32)
    batch_size = _tf.shape(tags)[0]
    max_length = _tf.shape(tags)[1]

    batch_ids = _tf.tile(_tf.range(batch_size)[:, None], [1, max_length])
    time_ids = _tf.tile(_tf.range(max_length)[None, :], [batch_size, 1])
    unary_indices = _tf.stack([batch_ids, time_ids, tags], axis=-1)
    unary_scores = _tf.gather_nd(emissions, unary_indices)
    unary_mask = _tf.sequence_mask(lengths, maxlen=max_length, dtype=emissions.dtype)
    unary_score = _tf.reduce_sum(unary_scores * unary_mask, axis=1)

    pair_indices = _tf.stack([tags[:, :-1], tags[:, 1:]], axis=-1)
    transition_scores = _tf.gather_nd(transitions, pair_indices)
    transition_lengths = _tf.maximum(lengths - 1, 0)
    transition_mask = _tf.sequence_mask(
        transition_lengths, maxlen=_tf.maximum(max_length - 1, 0), dtype=emissions.dtype
    )
    transition_score = _tf.reduce_sum(transition_scores * transition_mask, axis=1)
    return unary_score + transition_score


def crf_log_norm(
    emissions: "tf.Tensor",
    lengths: "tf.Tensor",
    transitions: "tf.Tensor",
) -> "tf.Tensor":
    _tf = _require_tensorflow()
    emissions = _tf.convert_to_tensor(emissions)
    lengths = _tf.cast(lengths, _tf.int32)
    max_length = _tf.shape(emissions)[1]
    alpha = emissions[:, 0, :]

    def condition(time: "tf.Tensor", _: "tf.Tensor") -> "tf.Tensor":
        return time < max_length

    def body(time: "tf.Tensor", current_alpha: "tf.Tensor") -> tuple["tf.Tensor", "tf.Tensor"]:
        scores = (
            current_alpha[:, :, None]
            + transitions[None, :, :]
            + emissions[:, time, None, :]
        )
        next_alpha = _tf.reduce_logsumexp(scores, axis=1)
        active = time < lengths
        current_alpha = _tf.where(active[:, None], next_alpha, current_alpha)
        return time + 1, current_alpha

    _, alpha = _tf.while_loop(
        condition,
        body,
        loop_vars=(_tf.constant(1), alpha),
        parallel_iterations=32,
    )
    return _tf.reduce_logsumexp(alpha, axis=1)


def crf_log_likelihood(
    emissions: "tf.Tensor",
    tags: "tf.Tensor",
    lengths: "tf.Tensor",
    transitions: "tf.Tensor",
) -> "tf.Tensor":
    return crf_sequence_score(emissions, tags, lengths, transitions) - crf_log_norm(
        emissions, lengths, transitions
    )


def crf_negative_log_likelihood(
    emissions: "tf.Tensor",
    tags: "tf.Tensor",
    lengths: "tf.Tensor",
    transitions: "tf.Tensor",
) -> "tf.Tensor":
    return -_require_tensorflow().reduce_mean(crf_log_likelihood(emissions, tags, lengths, transitions))


def viterbi_decode_numpy(
    emissions: np.ndarray,
    transitions: np.ndarray,
    lengths: Sequence[int],
    start_scores: np.ndarray | None = None,
) -> list[list[int]]:
    """Batch Viterbi decoding implemented in NumPy for transparent inference."""
    emissions = np.asarray(emissions, dtype=np.float64)
    transitions = np.asarray(transitions, dtype=np.float64)
    decoded: list[list[int]] = []
    for sequence_emissions, length in zip(emissions, lengths):
        length = int(length)
        if length <= 0:
            decoded.append([])
            continue
        score = sequence_emissions[0].copy()
        if start_scores is not None:
            score += start_scores
        backpointers: list[np.ndarray] = []
        for time in range(1, length):
            candidate = score[:, None] + transitions
            pointer = np.argmax(candidate, axis=0)
            score = candidate[pointer, np.arange(candidate.shape[1])] + sequence_emissions[time]
            backpointers.append(pointer)
        best_last = int(np.argmax(score))
        path = [best_last]
        for pointer in reversed(backpointers):
            best_last = int(pointer[best_last])
            path.append(best_last)
        decoded.append(list(reversed(path)))
    return decoded


def build_bio_constraints(
    index_to_tag: dict[int, str],
    invalid_penalty: float = -10_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return transition and start scores that block invalid BIO I-* positions."""
    size = len(index_to_tag)
    transitions = np.zeros((size, size), dtype=np.float32)
    start_scores = np.zeros(size, dtype=np.float32)

    for current_index, current_tag in index_to_tag.items():
        if current_tag.startswith("I-"):
            start_scores[current_index] = invalid_penalty

    for previous_index, previous_tag in index_to_tag.items():
        for current_index, current_tag in index_to_tag.items():
            if not current_tag.startswith("I-"):
                continue
            current_type = current_tag.split("-", 1)[1]
            valid_previous = previous_tag in {f"B-{current_type}", f"I-{current_type}"}
            if not valid_previous:
                transitions[previous_index, current_index] = invalid_penalty
    return transitions, start_scores



if tf is not None:
    class BiLSTMCRFTagger(tf.keras.Model):
        """BiLSTM emission network trained with linear-chain CRF likelihood."""

        def __init__(
            self,
            vocab_size: int,
            num_tags: int,
            max_sequence_length: int,
            pad_token_id: int = 0,
            embedding_dim: int = 100,
            lstm_units: int = 128,
            dense_units: int = 64,
            dropout_rate: float = 0.30,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.vocab_size = int(vocab_size)
            self.num_tags = int(num_tags)
            self.max_sequence_length = int(max_sequence_length)
            self.pad_token_id = int(pad_token_id)
            self.embedding_dim = int(embedding_dim)
            self.lstm_units = int(lstm_units)
            self.dense_units = int(dense_units)
            self.dropout_rate = float(dropout_rate)

            self.embedding = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                mask_zero=(self.pad_token_id == 0),
                name="token_embedding",
            )
            self.encoder = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True),
                name="bidirectional_lstm",
            )
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
            self.projection = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.dense_units, activation="relu"),
                name="token_projection",
            )
            self.emission_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.num_tags),
                name="tag_emissions",
            )
            self.transition_params = self.add_weight(
                name="crf_transitions",
                shape=(self.num_tags, self.num_tags),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.loss_tracker = tf.keras.metrics.Mean(name="loss")
            self.token_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")

        @property
        def metrics(self) -> list[tf.keras.metrics.Metric]:
            return [self.loss_tracker, self.token_accuracy]

        def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
            x = self.embedding(inputs)
            x = self.encoder(x, training=training)
            x = self.dropout(x, training=training)
            x = self.projection(x, training=training)
            return self.emission_layer(x, training=training)

        def _lengths(self, inputs: tf.Tensor) -> tf.Tensor:
            return tf.reduce_sum(
                tf.cast(tf.not_equal(inputs, self.pad_token_id), tf.int32), axis=1
            )

        def train_step(self, data):
            inputs, tags, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
            lengths = self._lengths(inputs)
            with tf.GradientTape() as tape:
                emissions = self(inputs, training=True)
                loss = crf_negative_log_likelihood(
                    emissions, tags, lengths, self.transition_params
                )
                if self.losses:
                    loss += tf.add_n(self.losses)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            mask = tf.sequence_mask(lengths, maxlen=tf.shape(tags)[1], dtype=tf.float32)
            self.loss_tracker.update_state(loss)
            self.token_accuracy.update_state(tags, emissions, sample_weight=mask)
            return {metric.name: metric.result() for metric in self.metrics}

        def test_step(self, data):
            inputs, tags, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
            lengths = self._lengths(inputs)
            emissions = self(inputs, training=False)
            loss = crf_negative_log_likelihood(
                emissions, tags, lengths, self.transition_params
            )
            mask = tf.sequence_mask(lengths, maxlen=tf.shape(tags)[1], dtype=tf.float32)
            self.loss_tracker.update_state(loss)
            self.token_accuracy.update_state(tags, emissions, sample_weight=mask)
            return {metric.name: metric.result() for metric in self.metrics}

        def decode(self, inputs: np.ndarray) -> list[list[int]]:
            emissions = self(inputs, training=False).numpy()
            lengths = np.sum(inputs != self.pad_token_id, axis=1)
            return viterbi_decode_numpy(
                emissions, self.transition_params.numpy(), lengths
            )

        def get_config(self) -> dict[str, object]:
            return {
                "vocab_size": self.vocab_size,
                "num_tags": self.num_tags,
                "max_sequence_length": self.max_sequence_length,
                "pad_token_id": self.pad_token_id,
                "embedding_dim": self.embedding_dim,
                "lstm_units": self.lstm_units,
                "dense_units": self.dense_units,
                "dropout_rate": self.dropout_rate,
                "name": self.name,
            }
else:
    class BiLSTMCRFTagger:
        def __init__(self, *args, **kwargs):
            _require_tensorflow()
