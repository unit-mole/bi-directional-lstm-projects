from __future__ import annotations


def build_siamese_bilstm(
    *,
    vocab_size: int,
    max_sequence_length: int,
    embedding_dimension: int = 128,
    bilstm_units: int = 64,
    dropout_rate: float = 0.30,
    learning_rate: float = 1e-3,
):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    @tf.keras.utils.register_keras_serializable(package="QAMatching")
    class AbsoluteDifference(layers.Layer):
        def call(self, inputs):
            return tf.abs(inputs[0] - inputs[1])

    @tf.keras.utils.register_keras_serializable(package="QAMatching")
    class ElementwiseProduct(layers.Layer):
        def call(self, inputs):
            return inputs[0] * inputs[1]

    encoder_input = layers.Input(shape=(max_sequence_length,), name="encoder_input")
    x = layers.Embedding(vocab_size, embedding_dimension, mask_zero=True, name="shared_embedding")(encoder_input)
    x = layers.Bidirectional(
        layers.LSTM(bilstm_units, return_sequences=True), name="shared_bilstm"
    )(x)
    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = layers.Dropout(dropout_rate, name="encoder_dropout")(x)
    encoder_output = layers.Dense(128, activation="relu", name="encoder_projection")(x)
    shared_encoder = Model(encoder_input, encoder_output, name="shared_encoder")

    text_a_input = layers.Input(shape=(max_sequence_length,), name="text_a_input")
    text_b_input = layers.Input(shape=(max_sequence_length,), name="text_b_input")
    vector_a = shared_encoder(text_a_input)
    vector_b = shared_encoder(text_b_input)

    difference = AbsoluteDifference(name="absolute_difference")([vector_a, vector_b])
    product = ElementwiseProduct(name="elementwise_product")([vector_a, vector_b])
    combined = layers.Concatenate(name="pair_features")([vector_a, vector_b, difference, product])
    combined = layers.Dense(128, activation="relu", name="dense_1")(combined)
    combined = layers.Dropout(0.30, name="dropout_1")(combined)
    combined = layers.Dense(64, activation="relu", name="dense_2")(combined)
    combined = layers.Dropout(0.20, name="dropout_2")(combined)
    output = layers.Dense(1, activation="sigmoid", name="match_probability")(combined)

    model = Model([text_a_input, text_b_input], output, name="qa_siamese_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model
