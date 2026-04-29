"""Contrastive loss for Siamese network training."""

import tensorflow as tf


def contrastive_loss(margin: float = 1.0):
    """
    Factory that returns a contrastive loss function.

    y=0 → similar pair  (distance should be small)
    y=1 → dissimilar pair (distance should be > margin)

    L = (1-y) * 0.5 * D²  +  y * 0.5 * max(0, margin - D)²
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        D = y_pred  # Euclidean distance from the Lambda layer
        similar_loss    = (1.0 - y_true) * 0.5 * tf.square(D)
        dissimilar_loss = y_true         * 0.5 * tf.square(tf.maximum(0.0, margin - D))
        return tf.reduce_mean(similar_loss + dissimilar_loss)

    loss_fn.__name__ = f"contrastive_loss_margin{margin}"
    return loss_fn
