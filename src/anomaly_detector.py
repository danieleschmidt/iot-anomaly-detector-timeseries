import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model # For loading the Keras model
import os 
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'src' is in PYTHONPATH or script is run from project root for this import
try:
    from src.data_preprocessor import TimeSeriesPreprocessor
except ModuleNotFoundError:
    from data_preprocessor import TimeSeriesPreprocessor


class AnomalyDetector:
    """
    Detects anomalies in new time series data using a pre-trained LSTM autoencoder model
    and a saved TimeSeriesPreprocessor.
    """
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Initializes the AnomalyDetector.

        Args:
            model_path (str): Path to the saved Keras autoencoder model (.keras file or .h5).
            preprocessor_path (str): Path to the saved TimeSeriesPreprocessor object (.pkl file).
        
        Raises:
            FileNotFoundError: If the model or preprocessor file is not found.
            Exception: If there's an error loading the model or preprocessor.
        """
        try:
            self.model = load_model(model_path)
            # print(f"Keras model loaded successfully from {model_path}") # Reduced verbosity
        except Exception as e:
            print(f"Error loading Keras model from {model_path}: {e}")
            raise
        
        try:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor: TimeSeriesPreprocessor = pickle.load(f)
            # print(f"TimeSeriesPreprocessor loaded successfully from {preprocessor_path}") # Reduced verbosity
        except FileNotFoundError:
            print(f"Error: Preprocessor file not found at {preprocessor_path}")
            raise
        except Exception as e:
            print(f"Error loading TimeSeriesPreprocessor from {preprocessor_path}: {e}")
            raise

    def predict_reconstruction_error(self, new_data_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts the reconstruction error for new data using the loaded autoencoder.

        Args:
            new_data_df (pd.DataFrame): New input DataFrame with the same features 
                                        as the training data (at least self.preprocessor.feature_cols).

        Returns:
            np.ndarray: A 1D NumPy array of reconstruction errors (MAE) for each sequence.
                        Returns an empty array if not enough data to form sequences.
        
        Raises:
            ValueError: If required feature_cols are missing from new_data_df.
            TypeError: If new_data_df is not a pandas DataFrame.
        """
        if not isinstance(new_data_df, pd.DataFrame):
            raise TypeError("Input new_data_df must be a pandas DataFrame.")

        df_proc = new_data_df.copy()

        missing_cols = [col for col in self.preprocessor.feature_cols if col not in df_proc.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required feature_cols: {', '.join(missing_cols)}")

        df_proc[self.preprocessor.feature_cols] = self.preprocessor.scaler.transform(
            df_proc[self.preprocessor.feature_cols]
        )
        
        original_sequences = self.preprocessor.create_sequences(df_proc)

        if original_sequences.shape[0] == 0:
            return np.array([]) 

        reconstructed_sequences = self.model.predict(original_sequences, verbose=0) 
        
        errors_per_sequence = tf.reduce_mean(
            tf.abs(tf.convert_to_tensor(original_sequences, dtype=tf.float32) - 
                   tf.convert_to_tensor(reconstructed_sequences, dtype=tf.float32)),
            axis=(1, 2) 
        ).numpy() 
        
        return errors_per_sequence

    def visualize_reconstruction_error(self, data_df_for_threshold_calc: pd.DataFrame):
        """
        Calculates reconstruction errors for the given DataFrame and plots their distribution.

        Args:
            data_df_for_threshold_calc (pd.DataFrame): DataFrame of (typically normal) data 
                                                       to calculate reconstruction errors from.
        """
        print("Calculating reconstruction errors for visualization...")
        errors = self.predict_reconstruction_error(data_df_for_threshold_calc)

        if errors.size == 0:
            print("No reconstruction errors to visualize (data might be too short for sequences).")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50, kde=True)
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error (MAE)')
        plt.ylabel('Density')
        plt.show()
        print("Reconstruction error visualization displayed.")


    def find_anomalies(self, new_data_df: pd.DataFrame, 
                       error_threshold: float = None, 
                       threshold_percentile: float = None, 
                       validation_data_for_threshold: pd.DataFrame = None) -> pd.DataFrame:
        """
        Identifies anomalies in new_data_df based on reconstruction error.
        The error threshold can be provided directly or calculated from validation data.

        Args:
            new_data_df (pd.DataFrame): New input DataFrame to check for anomalies.
            error_threshold (float, optional): Directly specified threshold. If None,
                                               threshold_percentile and validation_data_for_threshold
                                               must be provided.
            threshold_percentile (float, optional): Percentile (e.g., 0.95 for 95th) to use on
                                                    validation_data_for_threshold errors to set the threshold.
                                                    Range: 0.0 to 1.0. Ignored if error_threshold is set.
            validation_data_for_threshold (pd.DataFrame, optional): DataFrame of 'normal' data used to
                                                                    calculate the error_threshold if not
                                                                    directly provided. Ignored if error_threshold is set.

        Returns:
            pd.DataFrame: A sub-DataFrame of new_data_df containing the rows
                          that are identified as the start of an anomalous sequence.
                          Includes a 'reconstruction_error' column.
                          Returns an empty DataFrame if no anomalies are found or
                          if not enough data to form sequences.
        
        Raises:
            ValueError: If error_threshold is None and either threshold_percentile or
                        validation_data_for_threshold is also None.
                        Or if validation_data_for_threshold yields no errors.
                        Or if threshold_percentile is not between 0.0 and 1.0.
        """
        calculated_threshold = None
        if error_threshold is None:
            if threshold_percentile is None or validation_data_for_threshold is None:
                raise ValueError("If error_threshold is not provided, both threshold_percentile "
                                 "and validation_data_for_threshold must be provided.")
            if not (0.0 <= threshold_percentile <= 1.0):
                raise ValueError("threshold_percentile must be between 0.0 and 1.0.")
            
            print(f"Calculating dynamic error threshold using {threshold_percentile*100:.1f}th percentile of validation data errors...")
            validation_errors = self.predict_reconstruction_error(validation_data_for_threshold)
            if validation_errors.size == 0:
                raise ValueError("Could not determine threshold: validation_data_for_threshold "
                                 "yielded no reconstruction errors (it might be too short).")
            
            calculated_threshold = np.percentile(validation_errors, threshold_percentile * 100)
            print(f"Determined error threshold: {calculated_threshold:.4f}")
        else:
            calculated_threshold = error_threshold
            print(f"Using provided error threshold: {calculated_threshold:.4f}")


        reconstruction_errors = self.predict_reconstruction_error(new_data_df)

        # Define columns for the empty DataFrame, including the reconstruction error
        output_columns = list(new_data_df.columns) + ['reconstruction_error']
        if reconstruction_errors.size == 0:
            return pd.DataFrame(columns=output_columns)

        anomalous_sequence_indices = np.where(reconstruction_errors > calculated_threshold)[0]
        
        if anomalous_sequence_indices.size > 0:
            anomalous_df = new_data_df.iloc[anomalous_sequence_indices].copy()
            anomalous_df['reconstruction_error'] = reconstruction_errors[anomalous_sequence_indices]
            return anomalous_df
        else:
            return pd.DataFrame(columns=output_columns)


def plot_anomalies(original_df: pd.DataFrame, anomalous_df: pd.DataFrame, feature_to_plot: str):
    """
    Plots a feature from the original DataFrame and highlights anomalies.

    Args:
        original_df (pd.DataFrame): The original DataFrame with time series data.
                                     Must have a usable index for plotting (e.g., numeric or datetime).
        anomalous_df (pd.DataFrame): DataFrame containing anomalous data points (output of find_anomalies).
                                     Must have an index compatible with original_df and contain feature_to_plot.
        feature_to_plot (str): The name of the column in original_df to plot.
    
    Raises:
        ValueError: If feature_to_plot is not in original_df.columns.
    """
    if feature_to_plot not in original_df.columns:
        raise ValueError(f"Feature '{feature_to_plot}' not found in original_df columns.")

    plt.figure(figsize=(12, 6))
    plt.plot(original_df.index, original_df[feature_to_plot], label=f'Normal {feature_to_plot}', zorder=1)
    
    if not anomalous_df.empty:
        if feature_to_plot not in anomalous_df.columns:
             print(f"Warning: Feature '{feature_to_plot}' not in anomalous_df. Anomalies will be marked based on original_df values at anomalous indices.")
             # Check if indices from anomalous_df are present in original_df
             valid_anomalous_indices = anomalous_df.index[anomalous_df.index.isin(original_df.index)]
             if not valid_anomalous_indices.empty:
                 plt.scatter(valid_anomalous_indices, original_df.loc[valid_anomalous_indices, feature_to_plot], 
                             color='red', label='Anomaly', marker='o', s=50, zorder=2)
             else:
                 print("Warning: No valid anomalous indices found in original_df to plot.")
        else: # feature_to_plot is in anomalous_df (preferred)
             plt.scatter(anomalous_df.index, anomalous_df[feature_to_plot], 
                        color='red', label='Anomaly', marker='o', s=50, zorder=2)
    else:
        print("No anomalies to plot.")


    plt.title(f'Anomaly Detection in {feature_to_plot}')
    plt.xlabel('Time/Index')
    plt.ylabel(f'{feature_to_plot} Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
    print(f"Anomaly plot for '{feature_to_plot}' displayed.")


if __name__ == '__main__':
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR_EXAMPLE = os.path.dirname(CURRENT_SCRIPT_DIR) 
    SAVED_MODELS_DIR_EXAMPLE = os.path.join(BASE_DIR_EXAMPLE, 'saved_models')
    MODEL_PATH_EXAMPLE = os.path.join(SAVED_MODELS_DIR_EXAMPLE, 'lstm_autoencoder.keras')
    PREPROCESSOR_PATH_EXAMPLE = os.path.join(SAVED_MODELS_DIR_EXAMPLE, 'preprocessor.pkl')

    print(f"--- AnomalyDetector Enhanced Example Usage ---")
    print(f"Base directory determined as: {BASE_DIR_EXAMPLE}")
    print(f"Attempting to use model from: {MODEL_PATH_EXAMPLE}")
    print(f"Attempting to use preprocessor from: {PREPROCESSOR_PATH_EXAMPLE}")

    if not os.path.exists(SAVED_MODELS_DIR_EXAMPLE):
        os.makedirs(SAVED_MODELS_DIR_EXAMPLE)
        print(f"Created directory: {SAVED_MODELS_DIR_EXAMPLE}")

    dummy_feature_cols_main = ['sensor1', 'sensor2', 'sensor3']
    dummy_window_size_main = 5
    
    if not os.path.exists(PREPROCESSOR_PATH_EXAMPLE):
        print(f"Warning: Preprocessor at {PREPROCESSOR_PATH_EXAMPLE} not found. Creating a basic one for example.")
        dummy_preprocessor_main = TimeSeriesPreprocessor(window_size=dummy_window_size_main, feature_cols=dummy_feature_cols_main)
        dummy_scaler_data_main = pd.DataFrame(np.random.uniform(0, 100, size=(20, len(dummy_feature_cols_main))), columns=dummy_feature_cols_main)
        dummy_preprocessor_main.fit_transform_scale(dummy_scaler_data_main)
        with open(PREPROCESSOR_PATH_EXAMPLE, 'wb') as f_dummy_prep_main:
            pickle.dump(dummy_preprocessor_main, f_dummy_prep_main)
        print(f"Saved a dummy preprocessor to {PREPROCESSOR_PATH_EXAMPLE}")

    if not os.path.exists(MODEL_PATH_EXAMPLE):
        print(f"Warning: Model at {MODEL_PATH_EXAMPLE} not found. Creating a basic one for example.")
        try:
            from src.autoencoder_model import build_lstm_autoencoder
        except ModuleNotFoundError:
            from autoencoder_model import build_lstm_autoencoder 
        
        dummy_input_shape_main = (dummy_window_size_main, len(dummy_feature_cols_main))
        dummy_model_main = build_lstm_autoencoder(dummy_input_shape_main, [32, 16], [16, 32], 'sigmoid') 
        dummy_train_data = np.random.rand(50, dummy_window_size_main, len(dummy_feature_cols_main))
        dummy_model_main.fit(dummy_train_data, dummy_train_data, epochs=3, batch_size=4, verbose=0)
        dummy_model_main.save(MODEL_PATH_EXAMPLE)
        print(f"Saved a dummy model to {MODEL_PATH_EXAMPLE}")
    
    try:
        print("\nInitializing AnomalyDetector...")
        detector = AnomalyDetector(model_path=MODEL_PATH_EXAMPLE, preprocessor_path=PREPROCESSOR_PATH_EXAMPLE)

        total_rows = 50
        val_split_idx = 25
        
        s1_val = np.random.uniform(10, 30, size=val_split_idx)
        s2_val = np.random.uniform(15, 25, size=val_split_idx)
        s3_val = np.random.uniform(20, 30, size=val_split_idx)
        validation_df = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=val_split_idx, freq='T')),
            'sensor1': s1_val, 'sensor2': s2_val, 'sensor3': s3_val
        }).set_index('timestamp') # Use timestamp as index for plotting

        s1_new = np.random.uniform(10, 32, size=(total_rows - val_split_idx))
        s2_new = np.random.uniform(14, 26, size=(total_rows - val_split_idx))
        s3_new = np.random.uniform(18, 32, size=(total_rows - val_split_idx))
        sample_new_data_main = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start=validation_df.index[-1] + pd.Timedelta(minutes=1), periods=(total_rows - val_split_idx), freq='T')),
            'sensor1': s1_new, 'sensor2': s2_new, 'sensor3': s3_new,
        }).set_index('timestamp') # Use timestamp as index

        anomaly_start_time = sample_new_data_main.index[10]
        anomaly_end_time = sample_new_data_main.index[12]
        sample_new_data_main.loc[anomaly_start_time:anomaly_end_time, ['sensor1', 'sensor2']] = 150.0 
        
        print("\nVisualizing reconstruction errors on validation data (assumed normal)...")
        detector.visualize_reconstruction_error(validation_df)

        percentile_to_use = 0.95
        print(f"\nFinding anomalies using {percentile_to_use*100:.0f}th percentile threshold from validation data...")
        anomalous_data_percentile = detector.find_anomalies(
            sample_new_data_main, 
            threshold_percentile=percentile_to_use,
            validation_data_for_threshold=validation_df
        )
        print(f"Anomalous data points (percentile-based threshold):")
        if not anomalous_data_percentile.empty:
            print(anomalous_data_percentile)
        else:
            print("No anomalies found with percentile-based threshold.")
            
        fixed_threshold = 0.05 
        print(f"\nFinding anomalies using fixed threshold: {fixed_threshold}...")
        anomalous_data_fixed = detector.find_anomalies(
            sample_new_data_main, 
            error_threshold=fixed_threshold
        )
        print(f"Anomalous data points (fixed threshold):")
        if not anomalous_data_fixed.empty:
            print(anomalous_data_fixed)
        else:
            print("No anomalies found with fixed threshold.")
            
        anomalies_to_plot_df = anomalous_data_percentile if not anomalous_data_percentile.empty else anomalous_data_fixed
        
        print("\nPlotting anomalies for 'sensor1' on new data...")
        plot_anomalies(sample_new_data_main, anomalies_to_plot_df, 'sensor1')
        
        print("\nPlotting anomalies for 'sensor2' on new data...")
        plot_anomalies(sample_new_data_main, anomalies_to_plot_df, 'sensor2')

        print("--- AnomalyDetector Enhanced Example Usage Completed ---")

    except Exception as e_main:
        print(f"An error occurred during the AnomalyDetector example: {e_main}")
        import traceback
        traceback.print_exc()
