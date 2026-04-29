"""Siamese network model factory for Stage 6."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class L2Normalize(layers.Layer):
    """L2-normalize embeddings to the unit sphere along axis=-1."""

    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)

    def get_config(self):
        return super().get_config()


class EuclideanDistance(layers.Layer):
    """Euclidean distance between two embedding vectors, shape (batch, 1)."""

    def call(self, inputs):
        a, b = inputs
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True) + 1e-9)

    def get_config(self):
        return super().get_config()


def build_shared_cnn(
    window_size: int,
    n_features: int,
    embedding_dim: int = 128,
) -> keras.Model:
    """
    Stage 4 simple CNN with the regression head replaced by a metric embedding.

    Architecture (shared subnet):
        Input(window_size, n_features)
        → Conv1D(64, k=3, causal, relu)
        → MaxPooling1D(2)
        → Conv1D(32, k=3, causal, relu)
        → GlobalAveragePooling1D
        → Dense(32, relu)
        → Dense(embedding_dim, relu)   ← embedding vector
        → L2Normalize                  ← unit-sphere projection
    """
    inputs = layers.Input(shape=(window_size, n_features), name="window_input")

    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal", name="conv1")(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="causal", name="conv2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dense(embedding_dim, activation="relu", name="embedding")(x)
    embeddings = L2Normalize(name="l2_normalize")(x)

    return keras.Model(inputs, embeddings, name="SharedCNN")


def build_siamese(
    shared_cnn: keras.Model,
    window_size: int,
    n_features: int,
) -> keras.Model:
    """
    Full Siamese network: two branches sharing weights, outputting Euclidean distance.

        Input_A ──┐
                  ├─ shared_cnn ─→ embedding_A ─┐
        Input_B ──┘                              ├─→ EuclideanDistance → output
                   shared_cnn ─→ embedding_B ───┘
    """
    input_a = layers.Input(shape=(window_size, n_features), name="input_A")
    input_b = layers.Input(shape=(window_size, n_features), name="input_B")

    embedding_a = shared_cnn(input_a)
    embedding_b = shared_cnn(input_b)

    distance = EuclideanDistance(name="euclidean_distance")([embedding_a, embedding_b])

    return keras.Model(inputs=[input_a, input_b], outputs=distance, name="SiameseNetwork")


def distance_metric(threshold: float = 0.5):
    """Threshold-based binary accuracy: distance < threshold → same class (0)."""
    def binary_accuracy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_class = tf.cast(tf.squeeze(y_pred, axis=-1) >= threshold, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_class), tf.float32))

    binary_accuracy.__name__ = "binary_accuracy"
    return binary_accuracy
