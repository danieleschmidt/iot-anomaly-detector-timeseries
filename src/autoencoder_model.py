import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

def build_lstm_autoencoder(input_shape: tuple, 
                           lstm_units_encoder: list[int], 
                           lstm_units_decoder: list[int], 
                           dense_activation: str) -> Model:
    """
    Builds an LSTM autoencoder model.

    The model consists of an encoder and a decoder:
    - Encoder: Takes input sequences, processes them through LSTM layers.
               The final output of the encoder LSTM stack is a fixed-size vector 
               (the last hidden state of the final LSTM encoder layer).
               All intermediate LSTM layers in the encoder output full sequences.
    - RepeatVector: This vector is then repeated `input_shape[0]` (window_size) times 
                  to generate a sequence suitable for the decoder's input.
    - Decoder: Takes the repeated vector sequence, processes it through LSTM layers
               (all of which return full sequences), and reconstructs the original 
               input sequence shape using a TimeDistributed Dense layer.

    Args:
        input_shape (tuple): Shape of the input sequences, e.g., (window_size, n_features).
                             `input_shape[0]` is window_size (number of timesteps), 
                             `input_shape[1]` is n_features (number of features per timestep).
        lstm_units_encoder (list[int]): List of integers, where each integer is the number of
                                        units in an LSTM layer in the encoder. 
                                        The list length determines the number of LSTM layers.
        lstm_units_decoder (list[int]): List of integers, where each integer is the number of
                                        units in an LSTM layer in the decoder.
                                        The list length determines the number of LSTM layers.
        dense_activation (str): Activation function for the final TimeDistributed Dense layer
                                in the decoder, which reconstructs the features for each timestep.

    Returns:
        tensorflow.keras.models.Model: The compiled LSTM autoencoder model.
        
    Raises:
        ValueError: If lstm_units_encoder or lstm_units_decoder is empty.
    """
    if not lstm_units_encoder:
        raise ValueError("lstm_units_encoder list cannot be empty.")
    if not lstm_units_decoder:
        raise ValueError("lstm_units_decoder list cannot be empty.")

    # Define Input layer
    inputs = Input(shape=input_shape)
    
    # --- Encoder ---
    encoded_signal = inputs
    num_encoder_layers = len(lstm_units_encoder)
    for i, units in enumerate(lstm_units_encoder):
        is_last_layer_in_encoder_stack = (i == num_encoder_layers - 1)
        # The final LSTM layer in the encoder stack must output only its last hidden state (a 2D tensor)
        # to be compatible with the RepeatVector layer that follows.
        # All intermediate LSTM layers must return sequences (a 3D tensor) for the next LSTM layer.
        return_sequences_flag = not is_last_layer_in_encoder_stack
        
        encoded_signal = LSTM(units, activation='relu', return_sequences=return_sequences_flag)(encoded_signal)

    # At this point, encoded_signal is the output of the last LSTM encoder layer (a 2D tensor).
    
    # RepeatVector layer
    # This layer repeats the final encoded state (a 2D vector) `input_shape[0]` times,
    # effectively creating a 3D tensor where each timestep has the same feature vector.
    # This prepares the encoded representation to be processed by the decoder's LSTMs.
    repeated_encoded_signal = RepeatVector(input_shape[0])(encoded_signal)
    
    # --- Decoder ---
    decoded_signal = repeated_encoded_signal # Start with the output of RepeatVector
    
    # All LSTM layers in the decoder must have return_sequences=True to output full sequences
    # for the next layer or for the final TimeDistributed Dense layer.
    for units in lstm_units_decoder:
        decoded_signal = LSTM(units, activation='relu', return_sequences=True)(decoded_signal)
        
    # TimeDistributed Dense layer to reconstruct the input shape feature-wise for each timestep.
    # input_shape[1] corresponds to n_features.
    output_sequence = TimeDistributed(Dense(input_shape[1], activation=dense_activation))(decoded_signal)
    
    # Create the autoencoder model
    autoencoder_model = Model(inputs, output_sequence)
    
    # Compile the model
    autoencoder_model.compile(optimizer='adam', loss='mae') # mae for mean absolute error
    
    return autoencoder_model

if __name__ == '__main__':
    # Example Usage:
    window_size_example = 10
    n_features_example = 3
    
    print("Building model with multiple encoder layers:")
    encoder_units_multiple = [64, 32] 
    decoder_units_multiple = [32, 64] 
    activation_fn_example = 'sigmoid' # Use 'sigmoid' if data is scaled to [0,1]

    model_multi_layer_encoder = build_lstm_autoencoder(
        input_shape=(window_size_example, n_features_example),
        lstm_units_encoder=encoder_units_multiple,
        lstm_units_decoder=decoder_units_multiple,
        dense_activation=activation_fn_example
    )
    model_multi_layer_encoder.summary()

    print("\\nBuilding model with a single encoder layer:")
    encoder_units_single = [50] 
    decoder_units_single = [50] 
    
    model_single_layer_encoder = build_lstm_autoencoder(
        input_shape=(window_size_example, n_features_example),
        lstm_units_encoder=encoder_units_single,
        lstm_units_decoder=decoder_units_single,
        dense_activation=activation_fn_example
    )
    model_single_layer_encoder.summary()
    
    # Test error handling for empty layer unit lists
    try:
        build_lstm_autoencoder((10,1), [], [32], 'sigmoid')
    except ValueError as e:
        print(f"\\nCaught expected error for empty encoder units: {e}")

    try:
        build_lstm_autoencoder((10,1), [32], [], 'sigmoid')
    except ValueError as e:
        print(f"\\nCaught expected error for empty decoder units: {e}")

    # Example with 'linear' activation for reconstruction if data is not scaled (e.g. raw values)
    print("\\nBuilding model with linear output activation:")
    model_linear_reconstruction = build_lstm_autoencoder(
        input_shape=(window_size_example, n_features_example),
        lstm_units_encoder=encoder_units_single, # Using single encoder layer for this example
        lstm_units_decoder=decoder_units_single, # Using single decoder layer
        dense_activation='linear' 
    )
    model_linear_reconstruction.summary()
