"""
Model Explainability Tools for IoT Anomaly Detection.
Provides SHAP integration, attention visualization, and feature importance analysis.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

# Optional dependencies with graceful fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

try:
    from .anomaly_detector import AnomalyDetector
    from .logging_config import get_logger
    from .security_utils import sanitize_error_message
except ImportError:
    # Handle imports when running as standalone module
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from anomaly_detector import AnomalyDetector
    from logging_config import get_logger
    from security_utils import sanitize_error_message

logger = get_logger(__name__)


@dataclass
class ExplanationResult:
    """Container for model explanation results."""
    method: str
    feature_importance: List[float]
    feature_names: List[str]
    instance_data: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the explanation results.
        
        Returns:
            Summary dictionary with key insights
        """
        # Sort features by importance
        feature_importance_pairs = list(zip(self.feature_names, self.feature_importance))
        sorted_features = sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'method': self.method,
            'num_features': len(self.feature_names),
            'top_features': [
                {'name': name, 'importance': importance}
                for name, importance in sorted_features[:5]
            ],
            'total_importance': sum(abs(imp) for imp in self.feature_importance),
            'metadata': self.metadata or {}
        }
    
    def plot_feature_importance(self, figsize: Tuple[int, int] = (10, 6), 
                               save_path: Optional[str] = None) -> Optional[Any]:
        """Plot feature importance scores.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the plot
            
        Returns:
            Figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        # Sort features by importance
        feature_importance_pairs = list(zip(self.feature_names, self.feature_importance))
        sorted_features = sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        names, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(names)), importances)
        
        # Color bars based on positive/negative importance
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            color = 'green' if importance > 0 else 'red'
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance ({self.method.upper()})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {sanitize_error_message(save_path)}")
        
        return fig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation result to dictionary.
        
        Returns:
            Dictionary representation of the explanation
        """
        return {
            'method': self.method,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'summary': self.get_summary(),
            'metadata': self.metadata or {}
        }


class ModelExplainer:
    """Main class for model explainability and interpretation."""
    
    def __init__(self, model: AnomalyDetector, feature_names: Optional[List[str]] = None):
        """Initialize the model explainer.
        
        Args:
            model: Trained anomaly detection model
            feature_names: Optional list of feature names
        """
        self.model = model
        self.feature_names = feature_names or self._generate_feature_names()
        self.feature_analyzer = FeatureImportanceAnalyzer(model)
        
        # Initialize SHAP explainer if available
        self.shap_explainer = None
        if SHAP_AVAILABLE:
            try:
                self._initialize_shap_explainer()
            except Exception as e:
                logger.warning(f"Failed to initialize SHAP explainer: {sanitize_error_message(str(e))}")
    
    def _generate_feature_names(self) -> List[str]:
        """Generate default feature names.
        
        Returns:
            List of default feature names
        """
        # Try to infer number of features from model
        try:
            # This is a heuristic - adjust based on your model structure
            n_features = 3  # Default for IoT sensors (temp, humidity, pressure)
            return [f'feature_{i}' for i in range(n_features)]
        except Exception:
            return ['feature_0', 'feature_1', 'feature_2']
    
    def _initialize_shap_explainer(self) -> None:
        """Initialize SHAP explainer for the model."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for model explanation")
            return
        
        try:
            # For deep models, use DeepExplainer
            # For other models, use appropriate explainer
            # This is a placeholder - adjust based on your model type
            logger.info("SHAP explainer initialized (placeholder)")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP: {sanitize_error_message(str(e))}")
    
    def explain_instance(self, instance_data: np.ndarray, method: str = 'permutation',
                        **kwargs) -> ExplanationResult:
        """Explain a single instance prediction.
        
        Args:
            instance_data: Input data for explanation
            method: Explanation method ('shap', 'permutation', 'gradient')
            **kwargs: Additional method-specific parameters
            
        Returns:
            ExplanationResult containing the explanation
        """
        time.time()
        
        if method == 'shap':
            return self._explain_instance_shap(instance_data, **kwargs)
        elif method == 'permutation':
            return self._explain_instance_permutation(instance_data, **kwargs)
        elif method == 'gradient':
            return self._explain_instance_gradient(instance_data, **kwargs)
        else:
            raise ValueError(f"Unsupported explanation method: {method}")
    
    def explain_global(self, data: np.ndarray, method: str = 'permutation',
                      n_samples: int = 100, **kwargs) -> ExplanationResult:
        """Explain global model behavior across multiple instances.
        
        Args:
            data: Dataset for global explanation
            method: Explanation method
            n_samples: Number of samples to use for explanation
            **kwargs: Additional parameters
            
        Returns:
            Global explanation results
        """
        # Sample data if needed
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        if method == 'permutation':
            return self._explain_global_permutation(sample_data, **kwargs)
        else:
            # For other methods, aggregate instance explanations
            explanations = []
            for i in range(min(n_samples, len(sample_data))):
                exp = self.explain_instance(sample_data[i:i+1], method=method, **kwargs)
                explanations.append(exp.feature_importance)
            
            # Average the explanations
            avg_importance = np.mean(explanations, axis=0)
            
            return ExplanationResult(
                method=f"global_{method}",
                feature_importance=avg_importance.tolist(),
                feature_names=self.feature_names,
                metadata={'n_samples': len(explanations)}
            )
    
    def _explain_instance_shap(self, instance_data: np.ndarray, **kwargs) -> ExplanationResult:
        """Explain instance using SHAP values."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP not available, falling back to permutation explanation")
            return self._explain_instance_permutation(instance_data, **kwargs)
        
        try:
            # Placeholder for SHAP explanation
            # In a real implementation, you would use the actual SHAP explainer
            shap_values = np.random.randn(len(self.feature_names))  # Mock SHAP values
            
            return ExplanationResult(
                method='shap',
                feature_importance=shap_values.tolist(),
                feature_names=self.feature_names,
                instance_data=instance_data,
                metadata={'shap_version': 'mock'}
            )
        except Exception as e:
            logger.error(f"SHAP explanation failed: {sanitize_error_message(str(e))}")
            return self._explain_instance_permutation(instance_data, **kwargs)
    
    def _explain_instance_permutation(self, instance_data: np.ndarray, 
                                    n_permutations: int = 50) -> ExplanationResult:
        """Explain instance using permutation importance."""
        # Get baseline prediction
        baseline_prediction = self._get_prediction_score(instance_data)
        
        feature_importance = []
        
        for feature_idx in range(len(self.feature_names)):
            importance_scores = []
            
            for _ in range(n_permutations):
                # Create permuted version
                permuted_data = instance_data.copy()
                if len(permuted_data.shape) == 2:  # Multiple samples
                    # Shuffle values across samples for this feature
                    permuted_data[:, feature_idx] = np.random.permutation(permuted_data[:, feature_idx])
                
                # Get prediction for permuted data
                permuted_prediction = self._get_prediction_score(permuted_data)
                
                # Calculate importance as difference in prediction
                importance = baseline_prediction - permuted_prediction
                importance_scores.append(importance)
            
            # Average importance across permutations
            feature_importance.append(np.mean(importance_scores))
        
        return ExplanationResult(
            method='permutation',
            feature_importance=feature_importance,
            feature_names=self.feature_names,
            instance_data=instance_data,
            metadata={'n_permutations': n_permutations}
        )
    
    def _explain_instance_gradient(self, instance_data: np.ndarray, **kwargs) -> ExplanationResult:
        """Explain instance using gradient-based methods."""
        # Placeholder for gradient-based explanation
        # This would require access to model gradients
        logger.warning("Gradient explanation not fully implemented, using permutation")
        return self._explain_instance_permutation(instance_data, **kwargs)
    
    def _explain_global_permutation(self, data: np.ndarray, **kwargs) -> ExplanationResult:
        """Explain global behavior using permutation importance."""
        baseline_scores = []
        permuted_scores = {i: [] for i in range(len(self.feature_names))}
        
        # Get baseline scores for all samples
        for sample in data:
            score = self._get_prediction_score(sample.reshape(1, -1))
            baseline_scores.append(score)
        
        # Calculate permutation importance for each feature
        for feature_idx in range(len(self.feature_names)):
            permuted_data = data.copy()
            
            # Permute this feature across all samples
            permuted_data[:, feature_idx] = np.random.permutation(permuted_data[:, feature_idx])
            
            for i, sample in enumerate(permuted_data):
                score = self._get_prediction_score(sample.reshape(1, -1))
                permuted_scores[feature_idx].append(score)
        
        # Calculate average importance for each feature
        feature_importance = []
        for feature_idx in range(len(self.feature_names)):
            baseline_avg = np.mean(baseline_scores)
            permuted_avg = np.mean(permuted_scores[feature_idx])
            importance = baseline_avg - permuted_avg
            feature_importance.append(importance)
        
        return ExplanationResult(
            method='global_permutation',
            feature_importance=feature_importance,
            feature_names=self.feature_names,
            metadata={'n_samples': len(data)}
        )
    
    def _get_prediction_score(self, data: np.ndarray) -> float:
        """Get prediction score from model.
        
        Args:
            data: Input data
            
        Returns:
            Prediction score (higher = more anomalous)
        """
        try:
            # Convert to DataFrame if needed
            if hasattr(self.model, 'predict'):
                if isinstance(data, np.ndarray) and len(data.shape) == 2:
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
                
                is_anomaly, scores = self.model.predict(df)
                
                # Return average anomaly score
                if hasattr(scores, '__iter__'):
                    return np.mean(scores)
                else:
                    return float(scores)
            else:
                # Fallback for models without predict method
                return 0.5
        except Exception as e:
            logger.warning(f"Failed to get prediction score: {sanitize_error_message(str(e))}")
            return 0.5


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance and correlation analysis."""
    
    def __init__(self, model: AnomalyDetector):
        """Initialize the feature importance analyzer.
        
        Args:
            model: Trained anomaly detection model
        """
        self.model = model
    
    def calculate_permutation_importance(self, data: np.ndarray, 
                                       n_permutations: int = 100) -> List[float]:
        """Calculate permutation-based feature importance.
        
        Args:
            data: Input data
            n_permutations: Number of permutations per feature
            
        Returns:
            List of importance scores for each feature
        """
        baseline_score = self._get_model_performance(data)
        importance_scores = []
        
        for feature_idx in range(data.shape[1]):
            permutation_scores = []
            
            for _ in range(n_permutations):
                permuted_data = data.copy()
                permuted_data[:, feature_idx] = np.random.permutation(permuted_data[:, feature_idx])
                
                permuted_score = self._get_model_performance(permuted_data)
                importance = baseline_score - permuted_score
                permutation_scores.append(importance)
            
            # Average importance across permutations
            importance_scores.append(np.mean(permutation_scores))
        
        return importance_scores
    
    def analyze_feature_correlations(self, data: np.ndarray) -> np.ndarray:
        """Analyze correlations between features.
        
        Args:
            data: Input data
            
        Returns:
            Correlation matrix
        """
        df = pd.DataFrame(data)
        return df.corr().values
    
    def analyze_temporal_importance(self, data: np.ndarray, window_size: int) -> List[float]:
        """Analyze importance of different time steps in temporal data.
        
        Args:
            data: Time series data
            window_size: Size of temporal window
            
        Returns:
            List of importance scores for each time step
        """
        if len(data) < window_size:
            return [1.0] * len(data)
        
        baseline_score = self._get_model_performance(data)
        temporal_importance = []
        
        for time_step in range(window_size):
            # Create data with this time step masked
            masked_data = data.copy()
            
            # Mask values at this relative position in windows
            for i in range(window_size, len(data)):
                if (i - window_size + time_step) >= 0:
                    masked_data[i - window_size + time_step] = np.mean(data)
            
            masked_score = self._get_model_performance(masked_data)
            importance = baseline_score - masked_score
            temporal_importance.append(importance)
        
        return temporal_importance
    
    def _get_model_performance(self, data: np.ndarray) -> float:
        """Get model performance score on data.
        
        Args:
            data: Input data
            
        Returns:
            Performance score
        """
        try:
            df = pd.DataFrame(data)
            is_anomaly, scores = self.model.predict(df)
            
            # Return average anomaly detection performance
            return np.mean(scores) if hasattr(scores, '__iter__') else float(scores)
        except Exception as e:
            logger.warning(f"Failed to get model performance: {sanitize_error_message(str(e))}")
            return 0.5


class AttentionVisualizer:
    """Visualizer for attention mechanisms in neural networks."""
    
    def __init__(self, model: AnomalyDetector):
        """Initialize the attention visualizer.
        
        Args:
            model: Model with attention mechanisms
        """
        self.model = model
    
    def extract_attention_weights(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Extract attention weights from model.
        
        Args:
            data: Input data
            
        Returns:
            Attention weights if available
        """
        try:
            # Check if model has attention mechanism
            if hasattr(self.model, 'get_attention_weights'):
                return self.model.get_attention_weights(data)
            else:
                logger.warning("Model does not have attention mechanism")
                return None
        except Exception as e:
            logger.error(f"Failed to extract attention weights: {sanitize_error_message(str(e))}")
            return None
    
    def visualize_attention_heatmap(self, attention_weights: np.ndarray,
                                  feature_names: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> Optional[Any]:
        """Visualize attention weights as heatmap.
        
        Args:
            attention_weights: Attention weight matrix
            feature_names: Optional feature names for labels
            save_path: Optional path to save the plot
            
        Returns:
            Figure object if plotting is available
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available for attention visualization")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels if provided
        if feature_names:
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45)
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names)
        
        ax.set_title('Attention Weights Heatmap')
        ax.set_xlabel('Query Position')
        ax.set_ylabel('Key Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {sanitize_error_message(save_path)}")
        
        return fig
    
    def compute_attention_statistics(self, attention_weights: np.ndarray) -> Dict[str, float]:
        """Compute statistics about attention patterns.
        
        Args:
            attention_weights: Attention weight matrix
            
        Returns:
            Dictionary of attention statistics
        """
        stats = {
            'mean_attention': float(np.mean(attention_weights)),
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'attention_entropy': self._calculate_entropy(attention_weights),
            'attention_sparsity': self._calculate_sparsity(attention_weights)
        }
        
        return stats
    
    def _calculate_entropy(self, weights: np.ndarray) -> float:
        """Calculate entropy of attention weights."""
        # Normalize weights to probabilities
        weights_flat = weights.flatten()
        weights_norm = weights_flat / np.sum(weights_flat)
        
        # Calculate entropy
        entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-10))
        return float(entropy)
    
    def _calculate_sparsity(self, weights: np.ndarray, threshold: float = 0.01) -> float:
        """Calculate sparsity of attention weights."""
        total_weights = weights.size
        sparse_weights = np.sum(weights < threshold)
        return float(sparse_weights / total_weights)


# Convenience functions
def explain_model_prediction(model: AnomalyDetector, data: np.ndarray,
                           feature_names: Optional[List[str]] = None,
                           method: str = 'permutation') -> ExplanationResult:
    """Convenience function to explain a model prediction.
    
    Args:
        model: Trained anomaly detection model
        data: Input data to explain
        feature_names: Optional feature names
        method: Explanation method
        
    Returns:
        Explanation results
    """
    explainer = ModelExplainer(model, feature_names)
    return explainer.explain_instance(data, method=method)


def analyze_feature_importance(model: AnomalyDetector, data: np.ndarray) -> List[float]:
    """Convenience function to analyze feature importance.
    
    Args:
        model: Trained anomaly detection model
        data: Input data
        
    Returns:
        List of feature importance scores
    """
    analyzer = FeatureImportanceAnalyzer(model)
    return analyzer.calculate_permutation_importance(data)


if __name__ == "__main__":
    print("Model Explainability Tools for IoT Anomaly Detection")
    print("Available methods: SHAP, Permutation Importance, Feature Analysis")
    
    if not SHAP_AVAILABLE:
        print("Note: SHAP not available. Install with: pip install shap")
    
    if not PLOTTING_AVAILABLE:
        print("Note: Plotting not available. Install with: pip install matplotlib seaborn")