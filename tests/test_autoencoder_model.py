"""Comprehensive tests for autoencoder_model module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Handle TensorFlow import gracefully for testing environments without TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    Model = None

from src.autoencoder_model import build_autoencoder


class TestBuildAutoencoder:
    """Test suite for build_autoencoder function."""
    
    @pytest.fixture
    def mock_tf_components(self):
        """Mock TensorFlow components for testing without TensorFlow."""
        if TF_AVAILABLE:
            return None
        
        mock_model = MagicMock()
        mock_model.summary.return_value = None
        mock_model.count_params.return_value = 12345
        mock_model.layers = [MagicMock() for _ in range(7)]  # Typical layer count
        
        mock_input = MagicMock()
        mock_lstm1 = MagicMock()
        mock_lstm2 = MagicMock()
        mock_encoded = MagicMock()
        mock_repeat = MagicMock()
        mock_lstm3 = MagicMock()
        mock_decoded = MagicMock()
        
        return {
            'model': mock_model,
            'input': mock_input,
            'lstm1': mock_lstm1,
            'lstm2': mock_lstm2,
            'encoded': mock_encoded,
            'repeat': mock_repeat,
            'lstm3': mock_lstm3,
            'decoded': mock_decoded
        }
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_default_params(self):
        """Test building autoencoder with default parameters."""
        input_shape = (30, 3)
        model = build_autoencoder(input_shape)
        
        assert isinstance(model, Model)
        assert model.input_shape == (None, 30, 3)
        assert model.output_shape == (None, 30, 3)
        
        # Check that the model was compiled
        assert model.optimizer is not None
        assert model.compiled_loss is not None
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_custom_params(self):
        """Test building autoencoder with custom parameters."""
        input_shape = (50, 5)
        latent_dim = 24
        lstm_units = 64
        
        model = build_autoencoder(input_shape, latent_dim, lstm_units)
        
        assert isinstance(model, Model)
        assert model.input_shape == (None, 50, 5)
        assert model.output_shape == (None, 50, 5)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_layer_structure(self):
        """Test that the autoencoder has the expected layer structure."""
        input_shape = (30, 3)
        model = build_autoencoder(input_shape)
        
        # The model should have specific layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        
        # Should contain Input, LSTM layers, RepeatVector, TimeDistributed
        assert 'InputLayer' in layer_types or 'Input' in str(layer_types)
        assert layer_types.count('LSTM') >= 3  # Three LSTM layers
        assert 'RepeatVector' in layer_types
        assert 'TimeDistributed' in layer_types
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_parameter_count(self):
        """Test that parameter count is reasonable for different configurations."""
        # Small model
        model_small = build_autoencoder((10, 2), latent_dim=8, lstm_units=16)
        params_small = model_small.count_params()
        
        # Larger model
        model_large = build_autoencoder((30, 5), latent_dim=32, lstm_units=64)
        params_large = model_large.count_params()
        
        # Larger model should have more parameters
        assert params_large > params_small
        assert params_small > 0
        assert params_large > 0
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_prediction_shape(self):
        """Test that the model produces outputs of the correct shape."""
        input_shape = (20, 4)
        model = build_autoencoder(input_shape)
        
        # Create dummy input data
        batch_size = 2
        dummy_input = np.random.randn(batch_size, *input_shape)
        
        # Test prediction
        output = model.predict(dummy_input, verbose=0)
        
        assert output.shape == (batch_size, *input_shape)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_compilation(self):
        """Test that the model is properly compiled."""
        input_shape = (30, 3)
        model = build_autoencoder(input_shape)
        
        # Check optimizer
        assert hasattr(model, 'optimizer')
        assert model.optimizer is not None
        
        # Check loss function
        assert hasattr(model, 'compiled_loss')
        assert model.compiled_loss is not None
        
        # Should be able to fit on dummy data
        dummy_x = np.random.randn(10, *input_shape)
        dummy_y = np.random.randn(10, *input_shape)
        
        # This should not raise an error
        history = model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
        assert 'loss' in history.history
    
    def test_build_autoencoder_input_validation(self):
        """Test input parameter validation."""
        # Test with various input shapes
        valid_shapes = [(10, 1), (30, 3), (50, 5), (100, 10)]
        
        for shape in valid_shapes:
            if TF_AVAILABLE:
                model = build_autoencoder(shape)
                assert model.input_shape == (None, *shape)
                assert model.output_shape == (None, *shape)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_edge_cases(self):
        """Test autoencoder building with edge case parameters."""
        # Minimum viable parameters
        model_min = build_autoencoder((1, 1), latent_dim=1, lstm_units=1)
        assert isinstance(model_min, Model)
        
        # Large parameters
        model_large = build_autoencoder((100, 20), latent_dim=128, lstm_units=256)
        assert isinstance(model_large, Model)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_build_autoencoder_different_latent_dims(self):
        """Test building autoencoders with different latent dimensions."""
        input_shape = (30, 3)
        latent_dims = [4, 8, 16, 32, 64]
        
        for latent_dim in latent_dims:
            model = build_autoencoder(input_shape, latent_dim=latent_dim)
            assert isinstance(model, Model)
            
            # Test with dummy data
            dummy_input = np.random.randn(5, *input_shape)
            output = model.predict(dummy_input, verbose=0)
            assert output.shape == (5, *input_shape)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available") 
    def test_build_autoencoder_different_lstm_units(self):
        """Test building autoencoders with different LSTM unit counts."""
        input_shape = (30, 3)
        lstm_units_list = [8, 16, 32, 64, 128]
        
        for lstm_units in lstm_units_list:
            model = build_autoencoder(input_shape, lstm_units=lstm_units)
            assert isinstance(model, Model)
            
            # Larger LSTM units should generally mean more parameters
            # (though this can vary based on other factors)
            params = model.count_params()
            assert params > 0
    
    @pytest.mark.skipif(TF_AVAILABLE, reason="Testing without TensorFlow")
    def test_build_autoencoder_no_tensorflow(self):
        """Test behavior when TensorFlow is not available."""
        # This test simulates the case where TensorFlow is not installed
        with patch.dict('sys.modules', {'tensorflow': None}):
            with pytest.raises((ImportError, AttributeError)):
                # This should fail because TensorFlow components are not available
                build_autoencoder((30, 3))


class TestAutoencoderModelIntegration:
    """Integration tests for the autoencoder model."""
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_autoencoder_training_workflow(self):
        """Test a complete training workflow with the autoencoder."""
        input_shape = (20, 3)
        model = build_autoencoder(input_shape, latent_dim=8, lstm_units=16)
        
        # Generate synthetic training data
        n_samples = 100
        X_train = np.random.randn(n_samples, *input_shape)
        # For autoencoder, target is the same as input
        y_train = X_train.copy()
        
        # Train the model for a few epochs
        history = model.fit(
            X_train, y_train,
            epochs=3,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        # Check that training completed
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) == 3
        
        # Test prediction
        X_test = np.random.randn(10, *input_shape)
        predictions = model.predict(X_test, verbose=0)
        
        assert predictions.shape == X_test.shape
        assert not np.array_equal(predictions, X_test)  # Should be different from input
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_autoencoder_reconstruction_error(self):
        """Test that the autoencoder can compute reconstruction errors."""
        input_shape = (15, 2)
        model = build_autoencoder(input_shape, latent_dim=4, lstm_units=8)
        
        # Generate test data
        X_test = np.random.randn(20, *input_shape)
        
        # Get reconstructions
        reconstructions = model.predict(X_test, verbose=0)
        
        # Compute reconstruction errors
        reconstruction_errors = np.mean(np.square(X_test - reconstructions), axis=(1, 2))
        
        assert reconstruction_errors.shape == (20,)
        assert np.all(reconstruction_errors >= 0)  # Errors should be non-negative
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_autoencoder_batch_processing(self):
        """Test that the autoencoder handles different batch sizes correctly."""
        input_shape = (25, 4)
        model = build_autoencoder(input_shape, latent_dim=12, lstm_units=24)
        
        batch_sizes = [1, 5, 10, 32]
        
        for batch_size in batch_sizes:
            X_batch = np.random.randn(batch_size, *input_shape)
            output = model.predict(X_batch, verbose=0)
            
            assert output.shape == (batch_size, *input_shape)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_autoencoder_save_load_workflow(self):
        """Test saving and loading the autoencoder model."""
        import tempfile
        import os
        
        input_shape = (20, 3)
        model = build_autoencoder(input_shape)
        
        # Train briefly
        X_dummy = np.random.randn(50, *input_shape)
        model.fit(X_dummy, X_dummy, epochs=2, verbose=0)
        
        # Test prediction before saving
        test_input = np.random.randn(5, *input_shape)
        original_output = model.predict(test_input, verbose=0)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, 'test_autoencoder.h5')
            model.save(model_path)
            
            # Load model
            loaded_model = tf.keras.models.load_model(model_path)
            
            # Test that loaded model produces same output
            loaded_output = loaded_model.predict(test_input, verbose=0)
            
            np.testing.assert_array_almost_equal(original_output, loaded_output, decimal=5)


class TestAutoencoderModelPerformance:
    """Performance-related tests for the autoencoder model."""
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_model_parameter_efficiency(self):
        """Test that parameter count scales reasonably with model size."""
        base_shape = (30, 3)
        
        # Test different latent dimensions
        latent_dims = [4, 8, 16, 32]
        param_counts = []
        
        for latent_dim in latent_dims:
            model = build_autoencoder(base_shape, latent_dim=latent_dim)
            param_counts.append(model.count_params())
        
        # Parameter count should generally increase with latent dimension
        # (though not always strictly monotonic due to model architecture)
        assert all(count > 0 for count in param_counts)
        assert max(param_counts) > min(param_counts)
    
    @pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
    def test_memory_usage_reasonable(self):
        """Test that models don't create excessive memory usage."""
        # Create a reasonably large model
        input_shape = (100, 10)
        model = build_autoencoder(input_shape, latent_dim=50, lstm_units=100)
        
        # Parameter count should be reasonable (less than 10M for this size)
        param_count = model.count_params()
        assert param_count < 10_000_000, f"Model has {param_count:,} parameters, which may be excessive"
        
        # Should be able to process data without memory errors
        test_data = np.random.randn(10, *input_shape)
        output = model.predict(test_data, verbose=0)
        assert output.shape == test_data.shape


if __name__ == "__main__":
    pytest.main([__file__])