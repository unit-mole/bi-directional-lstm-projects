import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from src.attention_layer import TemporalAttention


def test_attention_scores_sum_to_one():
    layer = TemporalAttention()
    inputs = tf.random.normal((2, 4, 8))
    context, scores = layer(inputs)
    assert context.shape == (2, 8)
    tf.debugging.assert_near(tf.reduce_sum(scores, axis=1), tf.ones((2,)))
