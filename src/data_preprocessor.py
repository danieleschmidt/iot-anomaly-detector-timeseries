import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    def __init__(self, window_size: int, feature_cols: list[str]):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(feature_cols, list) or not feature_cols or not all(isinstance(col, str) for col in feature_cols):
            raise ValueError("feature_cols must be a non-empty list of strings.")

        self.window_size: int = window_size
        self.feature_cols: list[str] = feature_cols
        self.scaler: MinMaxScaler | None = None

    def fit_transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing specified feature_cols: {', '.join(missing_cols)}")

        self.scaler = MinMaxScaler()
        # Work on a copy to preserve the original DataFrame and to store scaled features alongside others
        df_scaled = df.copy()

        # Fit the scaler and transform the data for the specified feature columns
        df_scaled[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])

        return df_scaled

    def create_sequences(self, df_scaled: pd.DataFrame) -> np.ndarray:
        if not isinstance(df_scaled, pd.DataFrame):
            raise ValueError("Input for create_sequences must be a pandas DataFrame.")

        # Validate that feature_cols are present in the input df_scaled
        missing_cols_in_df = [col for col in self.feature_cols if col not in df_scaled.columns]
        if missing_cols_in_df:
            raise ValueError(f"DataFrame for create_sequences is missing columns: {', '.join(missing_cols_in_df)}")

        # Extract data for sequencing using the feature_cols
        data_values = df_scaled[self.feature_cols].values

        n_rows = len(data_values)
        # n_features is the number of columns in feature_cols
        n_features = data_values.shape[1] if n_rows > 0 else len(self.feature_cols)

        sequences = []
        # Only create sequences if there's enough data for at least one window
        if n_rows >= self.window_size:
            for i in range(n_rows - self.window_size + 1):
                # Each sequence is (window_size, n_features)
                sequences.append(data_values[i:(i + self.window_size), :])

        if not sequences:
            # Return an empty 3D array with the correct shape if no sequences could be formed
            return np.array([]).reshape(0, self.window_size, n_features)

        return np.array(sequences) # Shape: (n_samples, window_size, n_features)

    def inverse_transform_scale(self, data_to_revert: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call fit_transform_scale first.")

        is_input_dataframe = isinstance(data_to_revert, pd.DataFrame)

        features_array_for_inverse: np.ndarray
        # If input is a DataFrame, we'll try to preserve its structure (other columns, index)
        output_df_template: pd.DataFrame | None = None

        if is_input_dataframe:
            # data_to_revert is asserted to be a DataFrame here by is_input_dataframe
            df_input = data_to_revert.copy() # type: ignore[union-attr]
            output_df_template = df_input # Use this copy to fill in reverted values

            # Validate that all feature_cols (which scaler was fitted on) are present
            missing_cols = [col for col in self.feature_cols if col not in df_input.columns]
            if missing_cols:
                raise ValueError(f"DataFrame for inverse_transform_scale is missing feature_cols: {', '.join(missing_cols)}")

            # Extract data from the feature_cols (in the order scaler expects, which is self.feature_cols)
            features_array_for_inverse = df_input[self.feature_cols].values

        elif isinstance(data_to_revert, np.ndarray):
            # Input is a NumPy array
            if data_to_revert.ndim == 2 and data_to_revert.shape[1] == len(self.feature_cols):
                features_array_for_inverse = data_to_revert
            elif data_to_revert.ndim == 3:
                 raise ValueError(
                    f"Input for inverse_transform_scale is a 3D NumPy array (shape {data_to_revert.shape}). "
                    f"This method expects a 2D array of features (n_samples, n_features={len(self.feature_cols)}). "
                    "If this array is the output of create_sequences, you may need to reshape it or select specific samples/timesteps."
                )
            else: # Incorrect shape or dimensions for NumPy array
                raise ValueError(
                    f"NumPy array input must be 2D and have {len(self.feature_cols)} columns (number of features). "
                    f"Received {data_to_revert.ndim}D array with shape {data_to_revert.shape}."
                )
        else: # Input is neither DataFrame nor NumPy array
            raise ValueError("Input for inverse_transform_scale must be a pandas DataFrame or a 2D NumPy array.")

        # Perform inverse transformation
        # Handle empty array case to prevent sklearn error (e.g., if features_array_for_inverse is empty)
        if features_array_for_inverse.shape[0] == 0:
            reverted_data_array = np.array([]).reshape(0, len(self.feature_cols))
        else:
            reverted_data_array = self.scaler.inverse_transform(features_array_for_inverse)

        # Construct and return the result as a DataFrame
        if output_df_template is not None: # Input was a DataFrame
            # Place reverted data back into the copy of the original DataFrame structure
            output_df_template[self.feature_cols] = reverted_data_array
            return output_df_template
        else: # Input was a NumPy array
            # Create a new DataFrame for the reverted data, using feature_cols for column names
            return pd.DataFrame(reverted_data_array, columns=self.feature_cols)
