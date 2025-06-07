import unittest
import pandas as pd
import numpy as np

# Adjust the import path if necessary, assuming 'src' is in the PYTHONPATH
# or tests are run from the project root.
from src.data_preprocessor import TimeSeriesPreprocessor

class TestCreateSequences(unittest.TestCase):

    def test_basic_sequence_creation(self):
        """Test sequence creation with a valid DataFrame."""
        feature_cols = ['feature1', 'feature2']
        window_size = 3
        # Note: TimeSeriesPreprocessor's create_sequences uses _fitted_feature_names if fit_transform_scale was called,
        # otherwise it falls back to self.feature_cols. For these tests, we are testing create_sequences
        # in isolation or as it would behave if fit_transform_scale was not called prior,
        # thus relying on self.feature_cols.
        preprocessor = TimeSeriesPreprocessor(window_size=window_size, feature_cols=feature_cols)

        data = {
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [1, 2, 3, 4, 5],
            'other_col': [100, 200, 300, 400, 500] # Should be ignored
        }
        df = pd.DataFrame(data)

        sequences = preprocessor.create_sequences(df)

        # Expected: (len(df) - window_size + 1, window_size, len(feature_cols))
        # (5 - 3 + 1, 3, 2) = (3, 3, 2)
        self.assertEqual(sequences.shape, (3, window_size, len(feature_cols)))

        # Verify content of the first sequence
        expected_first_seq_data = df[feature_cols].iloc[0:window_size].values
        np.testing.assert_array_equal(sequences[0], expected_first_seq_data)

        # Verify content of the last sequence
        expected_last_seq_data = df[feature_cols].iloc[2:2+window_size].values
        np.testing.assert_array_equal(sequences[-1], expected_last_seq_data)

    def test_dataframe_shorter_than_window_size(self):
        """Test with DataFrame shorter than window_size."""
        feature_cols = ['a', 'b']
        window_size = 5
        preprocessor = TimeSeriesPreprocessor(window_size=window_size, feature_cols=feature_cols)

        data = {'a': [1, 2, 3], 'b': [10, 20, 30]} # Length 3
        df_short = pd.DataFrame(data)

        sequences = preprocessor.create_sequences(df_short)

        # Expect an empty array with shape (0, window_size, num_features)
        self.assertEqual(sequences.shape, (0, window_size, len(feature_cols)))
        self.assertEqual(sequences.size, 0)

    def test_empty_dataframe(self):
        """Test with an empty DataFrame."""
        feature_cols = ['col1', 'col2']
        window_size = 3
        preprocessor = TimeSeriesPreprocessor(window_size=window_size, feature_cols=feature_cols)

        empty_df = pd.DataFrame(columns=feature_cols)

        sequences = preprocessor.create_sequences(empty_df)

        self.assertEqual(sequences.shape, (0, window_size, len(feature_cols)))
        self.assertEqual(sequences.size, 0)

    def test_create_sequences_missing_columns_in_df(self):
        """Test create_sequences when DataFrame is missing a feature_col."""
        feature_cols = ['feature1', 'feature2']
        window_size = 3
        preprocessor = TimeSeriesPreprocessor(window_size=window_size, feature_cols=feature_cols)

        # DataFrame missing 'feature2'
        data_missing_col = {'feature1': [1, 2, 3, 4, 5]}
        df_missing_col = pd.DataFrame(data_missing_col)

        with self.assertRaisesRegex(ValueError, "DataFrame for create_sequences is missing columns: feature2"):
            preprocessor.create_sequences(df_missing_col)

    def test_create_sequences_uses_feature_cols_defined_in_init(self):
        """
        Test that create_sequences uses self.feature_cols when _fitted_feature_names is not set
        (i.e., fit_transform_scale has not been called).
        """
        init_feature_cols = ['f1', 'f2']
        window_size = 2
        preprocessor = TimeSeriesPreprocessor(window_size=window_size, feature_cols=init_feature_cols)

        # DataFrame with more columns than specified in init_feature_cols
        data = {
            'f1': [10, 20, 30],
            'f2': [1, 2, 3],
            'f3_ignored': [100, 200, 300]
        }
        df_for_sequencing = pd.DataFrame(data)

        # Since fit_transform_scale was not called, preprocessor._fitted_feature_names is None.
        # create_sequences should use preprocessor.feature_cols (['f1', 'f2']).
        sequences = preprocessor.create_sequences(df_for_sequencing)

        # Expected shape: (num_samples, window_size, num_init_features)
        # num_samples = len(df) - window_size + 1 = 3 - 2 + 1 = 2
        # num_init_features = len(init_feature_cols) = 2
        self.assertEqual(sequences.shape, (2, window_size, len(init_feature_cols)))

        # Verify that only 'f1' and 'f2' were used for sequences
        expected_first_seq_f1_f2 = df_for_sequencing[init_feature_cols].iloc[0:window_size].values
        np.testing.assert_array_equal(sequences[0], expected_first_seq_f1_f2)

        # Further check: if preprocessor was instantiated with different feature_cols than present,
        # it should raise error if those are not in the DF passed to create_sequences.
        preprocessor_expects_other_cols = TimeSeriesPreprocessor(window_size=2, feature_cols=['x', 'y'])
        with self.assertRaisesRegex(ValueError, "DataFrame for create_sequences is missing columns: x, y"):
            preprocessor_expects_other_cols.create_sequences(df_for_sequencing) # df_for_sequencing has f1,f2,f3_ignored

if __name__ == '__main__':
    # This setup allows running tests directly if needed, ensuring src path.
    # However, standard test discovery (e.g., python -m unittest discover tests) is preferred.
    # import sys
    # import os
    # # Add project root to Python path to allow direct import of src.data_preprocessor
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)
    # from src.data_preprocessor import TimeSeriesPreprocessor # Re-import for clarity if running directly

    unittest.main(verbosity=2)
