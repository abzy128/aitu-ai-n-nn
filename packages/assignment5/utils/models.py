"""1D Convolutional Autoencoder for Jena weather sequences."""

from tensorflow import keras
from tensorflow.keras import layers


def build_conv_autoencoder(
    window_size: int,
    n_features: int,
    bottleneck_dim: int = 16,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Symmetric 1D Conv Autoencoder with a GlobalAveragePooling1D bottleneck.

    Encoder:
        Input(window_size, n_features)
        → Conv1D(64, k=3, relu, same) → MaxPooling1D(2)   # (12, 64)
        → Conv1D(32, k=3, relu, same) → MaxPooling1D(2)   # (6, 32)
        → Conv1D(16, k=3, relu, same)                      # (6, 16)
        → GlobalAveragePooling1D                           # (16,) ← bottleneck

    Decoder:
        → RepeatVector(window_size // 4)                   # (6, 16)
        → Conv1D(16, k=3, relu, same)
        → UpSampling1D(2)                                  # (12, 16)
        → Conv1D(32, k=3, relu, same)
        → UpSampling1D(2)                                  # (24, 32)
        → Conv1D(n_features, k=3, linear, same)            # (24, n_features)

    Trained clean-to-clean: autoencoder(X_clean) ≈ X_clean.
    """
    compressed_len = window_size // 4  # 6 for window_size=24

    inputs = layers.Input(shape=(window_size, n_features), name="encoder_input")

    # Encoder
    x = layers.Conv1D(64, 3, activation="relu", padding="same", name="enc_conv1")(inputs)
    x = layers.MaxPooling1D(2, name="enc_pool1")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.MaxPooling1D(2, name="enc_pool2")(x)
    x = layers.Conv1D(bottleneck_dim, 3, activation="relu", padding="same", name="enc_conv3")(x)
    bottleneck = layers.GlobalAveragePooling1D(name="bottleneck")(x)

    # Decoder
    x = layers.RepeatVector(compressed_len, name="dec_repeat")(bottleneck)
    x = layers.Conv1D(bottleneck_dim, 3, activation="relu", padding="same", name="dec_conv1")(x)
    x = layers.UpSampling1D(2, name="dec_upsample1")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same", name="dec_conv2")(x)
    x = layers.UpSampling1D(2, name="dec_upsample2")(x)
    outputs = layers.Conv1D(n_features, 3, activation="linear", padding="same", name="dec_output")(x)

    autoencoder = keras.Model(inputs, outputs, name="Autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder


def extract_encoder(autoencoder: keras.Model) -> keras.Model:
    """Extract the encoder as a standalone model (input → bottleneck vector)."""
    return keras.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("bottleneck").output,
        name="Encoder",
    )


def extract_decoder(autoencoder: keras.Model, bottleneck_dim: int = 16) -> keras.Model:
    """Extract the decoder as a standalone model (bottleneck vector → reconstructed window).

    Reuses the trained decoder layer weights from the full autoencoder.
    """
    decoder_input = keras.Input(shape=(bottleneck_dim,), name="decoder_input")
    x = decoder_input
    for name in ["dec_repeat", "dec_conv1", "dec_upsample1", "dec_conv2", "dec_upsample2", "dec_output"]:
        x = autoencoder.get_layer(name)(x)
    return keras.Model(decoder_input, x, name="Decoder")
