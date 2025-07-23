"""
Test cases for Model Explainability Tools.
Tests SHAP integration, attention visualization, and feature importance.
"""
import pytest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock SHAP since it may not be available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = MagicMock()

from model_explainability import (
    ModelExplainer,
    FeatureImportanceAnalyzer,
    AttentionVisualizer,
    ExplanationResult
)


class TestModelExplainer:
    """Test the main ModelExplainer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock anomaly detection model."""
        model = MagicMock()
        model.predict.return_value = ([True, False, True], [0.8, 0.2, 0.9])
        model.preprocessor = MagicMock()
        model.preprocessor.scaler = MagicMock()
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        return np.random.randn(100, 3)  # 100 time steps, 3 features
    
    def test_explainer_initialization(self, mock_model):
        """Test ModelExplainer initialization."""
        explainer = ModelExplainer(mock_model)
        assert explainer.model == mock_model
        assert explainer.feature_names is not None
    
    def test_explain_instance_shap(self, mock_model, sample_data):
        """Test SHAP-based instance explanation."""
        explainer = ModelExplainer(mock_model)
        
        with patch('model_explainability.shap') as mock_shap:
            # Mock SHAP explainer
            mock_explainer = MagicMock()
            mock_explainer.shap_values.return_value = np.random.randn(10, 3)
            mock_shap.DeepExplainer.return_value = mock_explainer
            
            # Test explanation
            result = explainer.explain_instance(sample_data[:10], method='shap')
            
            assert isinstance(result, ExplanationResult)
            assert result.method == 'shap'
            assert result.feature_importance is not None
    
    def test_explain_instance_permutation(self, mock_model, sample_data):
        """Test permutation-based instance explanation."""
        explainer = ModelExplainer(mock_model)
        
        # Mock model predictions for permutation test
        mock_model.predict.side_effect = [
            ([True, False], [0.8, 0.2]),  # Original prediction
            ([False, False], [0.3, 0.2]),  # Permuted feature 0
            ([True, True], [0.8, 0.7]),   # Permuted feature 1
            ([True, False], [0.8, 0.1]),  # Permuted feature 2
        ]
        
        result = explainer.explain_instance(sample_data[:2], method='permutation')
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'permutation'
        assert len(result.feature_importance) == sample_data.shape[1]
    
    def test_explain_instance_invalid_method(self, mock_model, sample_data):
        """Test explanation with invalid method."""
        explainer = ModelExplainer(mock_model)
        
        with pytest.raises(ValueError, match="Unsupported explanation method"):
            explainer.explain_instance(sample_data[:10], method='invalid')
    
    def test_explain_global_feature_importance(self, mock_model, sample_data):
        """Test global feature importance explanation."""
        explainer = ModelExplainer(mock_model)
        
        # Mock multiple predictions for global analysis
        mock_model.predict.return_value = ([True, False, True], [0.8, 0.2, 0.9])
        
        result = explainer.explain_global(sample_data, method='permutation', n_samples=10)
        
        assert isinstance(result, ExplanationResult)
        assert result.method == 'permutation'
        assert result.feature_importance is not None


class TestFeatureImportanceAnalyzer:
    """Test the FeatureImportanceAnalyzer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.predict.return_value = ([True, False, True], [0.8, 0.2, 0.9])
        return model
    
    @pytest.fixture
    def analyzer(self, mock_model):
        """Create FeatureImportanceAnalyzer instance."""
        return FeatureImportanceAnalyzer(mock_model)
    
    def test_permutation_importance(self, analyzer):
        """Test permutation importance calculation."""
        data = np.random.randn(50, 3)
        
        # Mock predictions for permutation test
        analyzer.model.predict.side_effect = [
            ([True] * 25, [0.8] * 25),  # Original
            ([False] * 25, [0.3] * 25), # Permuted feature 0
            ([True] * 25, [0.7] * 25),  # Permuted feature 1
            ([True] * 25, [0.8] * 25),  # Permuted feature 2
        ]
        
        importance_scores = analyzer.calculate_permutation_importance(data)
        
        assert len(importance_scores) == data.shape[1]
        assert all(isinstance(score, (int, float)) for score in importance_scores)
    
    def test_feature_correlation_analysis(self, analyzer):
        """Test feature correlation analysis."""
        data = np.random.randn(100, 4)
        
        correlations = analyzer.analyze_feature_correlations(data)
        
        assert correlations.shape == (4, 4)  # Correlation matrix
        assert np.allclose(np.diag(correlations), 1.0)  # Diagonal should be 1
    
    def test_temporal_importance(self, analyzer):
        """Test temporal feature importance analysis."""
        data = np.random.randn(100, 3)
        window_size = 10
        
        # Mock predictions for temporal analysis
        analyzer.model.predict.return_value = ([True] * 91, [0.8] * 91)
        
        temporal_importance = analyzer.analyze_temporal_importance(data, window_size)
        
        assert len(temporal_importance) == window_size
        assert all(isinstance(score, (int, float)) for score in temporal_importance)


class TestAttentionVisualizer:
    """Test the AttentionVisualizer class."""
    
    @pytest.fixture
    def mock_model_with_attention(self):
        """Create a mock model with attention weights."""
        model = MagicMock()
        # Mock attention weights
        model.get_attention_weights.return_value = np.random.randn(32, 10, 10)  # batch, seq, seq
        return model
    
    def test_extract_attention_weights(self, mock_model_with_attention):
        """Test attention weight extraction."""
        visualizer = AttentionVisualizer(mock_model_with_attention)
        data = np.random.randn(32, 10, 3)
        
        attention_weights = visualizer.extract_attention_weights(data)
        
        assert attention_weights.shape == (32, 10, 10)
    
    def test_visualize_attention_heatmap(self, mock_model_with_attention):
        """Test attention heatmap visualization."""
        visualizer = AttentionVisualizer(mock_model_with_attention)
        attention_weights = np.random.randn(10, 10)
        
        with patch('model_explainability.plt') as mock_plt:
            fig = visualizer.visualize_attention_heatmap(attention_weights)
            
            # Verify matplotlib functions were called
            mock_plt.figure.assert_called()
            mock_plt.imshow.assert_called()
    
    def test_attention_summary_statistics(self, mock_model_with_attention):
        """Test attention summary statistics."""
        visualizer = AttentionVisualizer(mock_model_with_attention)
        attention_weights = np.random.randn(32, 10, 10)
        
        stats = visualizer.compute_attention_statistics(attention_weights)
        
        assert 'mean_attention' in stats
        assert 'max_attention' in stats
        assert 'attention_entropy' in stats
        assert 'attention_sparsity' in stats


class TestExplanationResult:
    """Test the ExplanationResult class."""
    
    def test_explanation_result_creation(self):
        """Test ExplanationResult creation and attributes."""
        feature_importance = [0.3, 0.5, 0.2]
        
        result = ExplanationResult(
            method='shap',
            feature_importance=feature_importance,
            feature_names=['feature_0', 'feature_1', 'feature_2']
        )
        
        assert result.method == 'shap'
        assert result.feature_importance == feature_importance
        assert len(result.feature_names) == 3
    
    def test_explanation_result_summary(self):
        """Test explanation result summary generation."""
        result = ExplanationResult(
            method='permutation',
            feature_importance=[0.1, 0.8, 0.3],
            feature_names=['temp', 'pressure', 'humidity']
        )
        
        summary = result.get_summary()
        
        assert 'method' in summary
        assert 'top_features' in summary
        assert summary['top_features'][0]['name'] == 'pressure'  # Highest importance
    
    def test_explanation_result_visualization(self):
        """Test explanation result visualization."""
        result = ExplanationResult(
            method='shap',
            feature_importance=[0.2, 0.6, 0.4],
            feature_names=['feature_A', 'feature_B', 'feature_C']
        )
        
        with patch('model_explainability.plt') as mock_plt:
            fig = result.plot_feature_importance()
            
            # Verify plotting functions were called
            mock_plt.figure.assert_called()
            mock_plt.barh.assert_called()
    
    def test_explanation_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ExplanationResult(
            method='permutation',
            feature_importance=[0.1, 0.5, 0.3],
            feature_names=['x', 'y', 'z'],
            metadata={'n_samples': 100}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['method'] == 'permutation'
        assert 'feature_importance' in result_dict
        assert 'metadata' in result_dict


class TestIntegrationExplainability:
    """Test integration between explainability components."""
    
    def test_end_to_end_explanation_pipeline(self):
        """Test complete explanation pipeline."""
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = ([True, False, True], [0.8, 0.2, 0.9])
        
        # Create explainer
        explainer = ModelExplainer(mock_model)
        
        # Sample data
        data = np.random.randn(20, 3)
        
        # Test instance explanation
        instance_result = explainer.explain_instance(data[:1], method='permutation')
        assert isinstance(instance_result, ExplanationResult)
        
        # Test global explanation
        global_result = explainer.explain_global(data, method='permutation', n_samples=5)
        assert isinstance(global_result, ExplanationResult)
    
    def test_explainer_with_real_data_format(self):
        """Test explainer with realistic IoT data format."""
        # Mock model that expects DataFrame input
        mock_model = MagicMock()
        mock_model.predict.return_value = ([True, False], [0.8, 0.2])
        
        explainer = ModelExplainer(
            mock_model,
            feature_names=['temperature', 'humidity', 'pressure']
        )
        
        # Realistic IoT sensor data
        data = np.array([
            [25.5, 60.2, 1013.25],  # Normal reading
            [35.8, 45.1, 1008.90],  # Potential anomaly
        ])
        
        result = explainer.explain_instance(data, method='permutation')
        
        assert result.feature_names == ['temperature', 'humidity', 'pressure']
        assert len(result.feature_importance) == 3