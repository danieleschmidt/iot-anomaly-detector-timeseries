from tensorflow.keras import layers, models, losses


def build_autoencoder(input_shape, latent_dim=16, lstm_units=32):
    """Build a simple sequence autoencoder."""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    encoded = layers.LSTM(latent_dim)(x)
    x = layers.RepeatVector(input_shape[0])(encoded)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    decoded = layers.TimeDistributed(layers.Dense(input_shape[1]))(x)
    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    return model
