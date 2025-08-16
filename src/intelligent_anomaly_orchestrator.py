"""Intelligent Anomaly Detection Orchestrator.

This module provides a sophisticated orchestration system that coordinates
multiple anomaly detection approaches, manages model ensembles, and provides
intelligent decision-making for IoT anomaly detection.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from .anomaly_detector import AnomalyDetector
from .data_drift_detector import DataDriftDetector
from .logging_config import get_logger
from .model_explainability import ModelExplainability
from .quantum_hybrid_autoencoder import QuantumHybridAutoencoder
from .transformer_autoencoder import TransformerAutoencoder


class DetectionStrategy(Enum):
    """Anomaly detection strategy options."""
    ENSEMBLE_VOTING = "ensemble_voting"
    WEIGHTED_AVERAGE = "weighted_average"
    ADAPTIVE_SELECTION = "adaptive_selection"
    CONFIDENCE_BASED = "confidence_based"
    MULTI_SCALE = "multi_scale"


class AnomalyClass(Enum):
    """Classification of anomaly types."""
    POINT_ANOMALY = "point"
    CONTEXTUAL_ANOMALY = "contextual"
    COLLECTIVE_ANOMALY = "collective"
    DRIFT_ANOMALY = "drift"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result from anomaly detection."""
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    anomaly_class: AnomalyClass
    explanation: Dict[str, Any]
    timestamp: float
    model_contributions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for individual detection models."""
    name: str
    model_path: str
    weight: float = 1.0
    enabled: bool = True
    model_type: str = "autoencoder"
    threshold: float = 0.5
    confidence_threshold: float = 0.7


@dataclass
class OrchestrationConfig:
    """Configuration for the orchestration system."""
    strategy: DetectionStrategy = DetectionStrategy.ENSEMBLE_VOTING
    enable_adaptive_weights: bool = True
    enable_drift_detection: bool = True
    enable_explainability: bool = True
    consensus_threshold: float = 0.6
    min_models_agreement: int = 2
    performance_window_size: int = 1000
    auto_model_selection: bool = True


class IntelligentAnomalyOrchestrator:
    """Intelligent orchestrator for multi-model anomaly detection."""

    def __init__(
        self,
        models_config: List[ModelConfig],
        config: Optional[OrchestrationConfig] = None,
        max_workers: int = 8
    ):
        """Initialize the intelligent anomaly orchestrator.
        
        Args:
            models_config: List of model configurations
            config: Orchestration configuration
            max_workers: Maximum number of worker threads
        """
        self.logger = get_logger(__name__)
        self.config = config or OrchestrationConfig()
        self.max_workers = max_workers

        # Initialize models
        self.models: Dict[str, Any] = {}
        self.model_configs = {cfg.name: cfg for cfg in models_config}
        self.model_weights = {cfg.name: cfg.weight for cfg in models_config}
        self.model_performance = defaultdict(lambda: deque(maxlen=self.config.performance_window_size))

        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_lock = threading.RLock()

        # Performance tracking
        self.detection_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(list)

        # Components
        self.drift_detector = DataDriftDetector() if self.config.enable_drift_detection else None
        self.explainer = ModelExplainability() if self.config.enable_explainability else None

        # Initialize models
        self._initialize_models()

        self.logger.info(f"Initialized orchestrator with {len(self.models)} models")

    def _initialize_models(self) -> None:
        """Initialize all detection models."""
        for model_config in self.model_configs.values():
            if not model_config.enabled:
                continue

            try:
                if model_config.model_type == "autoencoder":
                    model = AnomalyDetector(model_config.model_path)
                elif model_config.model_type == "quantum_hybrid":
                    model = QuantumHybridAutoencoder(model_config.model_path)
                elif model_config.model_type == "transformer":
                    model = TransformerAutoencoder(model_config.model_path)
                else:
                    self.logger.warning(f"Unknown model type: {model_config.model_type}")
                    continue

                self.models[model_config.name] = model
                self.logger.info(f"Loaded model: {model_config.name}")

            except Exception as e:
                self.logger.error(f"Failed to load model {model_config.name}: {e}")

    async def detect_anomaly(
        self,
        data: Union[Dict[str, Any], pd.DataFrame, np.ndarray],
        strategy: Optional[DetectionStrategy] = None
    ) -> DetectionResult:
        """Detect anomalies using the orchestrated approach.
        
        Args:
            data: Input data for anomaly detection
            strategy: Detection strategy to use (overrides config)
            
        Returns:
            Comprehensive detection result
        """
        start_time = time.time()
        strategy = strategy or self.config.strategy

        try:
            # Convert data to standard format
            processed_data = self._preprocess_input(data)

            # Detect drift if enabled
            drift_detected = False
            if self.drift_detector and len(self.detection_history) > 100:
                historical_data = [d.metadata.get('features') for d in self.detection_history[-100:] if d.metadata.get('features') is not None]
                if historical_data:
                    drift_detected = self.drift_detector.detect_drift(
                        np.array(historical_data),
                        processed_data.reshape(1, -1) if processed_data.ndim == 1 else processed_data
                    )

            # Run detection on all enabled models concurrently
            model_futures = {}
            with self.model_lock:
                for name, model in self.models.items():
                    if self.model_configs[name].enabled:
                        future = self.executor.submit(self._run_model_detection, name, model, processed_data)
                        model_futures[name] = future

            # Collect results
            model_results = {}
            for name, future in model_futures.items():
                try:
                    result = await asyncio.get_event_loop().run_in_executor(None, future.result, 5.0)
                    model_results[name] = result
                except Exception as e:
                    self.logger.warning(f"Model {name} failed: {e}")
                    model_results[name] = {
                        'score': 0.0,
                        'confidence': 0.0,
                        'error': str(e)
                    }

            # Apply detection strategy
            final_result = self._apply_detection_strategy(strategy, model_results, drift_detected)

            # Add explanation if enabled
            if self.config.enable_explainability and self.explainer:
                try:
                    explanation = await self._generate_explanation(processed_data, model_results, final_result)
                    final_result.explanation.update(explanation)
                except Exception as e:
                    self.logger.warning(f"Explanation generation failed: {e}")

            # Update performance tracking
            self._update_performance_tracking(model_results, final_result)

            # Store in history
            final_result.metadata.update({
                'features': processed_data.tolist() if hasattr(processed_data, 'tolist') else processed_data,
                'processing_time_ms': (time.time() - start_time) * 1000,
                'drift_detected': drift_detected,
                'num_models': len(model_results)
            })

            self.detection_history.append(final_result)

            return final_result

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return DetectionResult(
                anomaly_score=0.0,
                is_anomaly=False,
                confidence=0.0,
                anomaly_class=AnomalyClass.UNKNOWN,
                explanation={'error': str(e)},
                timestamp=time.time()
            )

    def _preprocess_input(self, data: Union[Dict[str, Any], pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Preprocess input data to standard format."""
        if isinstance(data, dict):
            # Convert dict to array
            return np.array(list(data.values()))
        elif isinstance(data, pd.DataFrame):
            return data.values.flatten() if data.size > 0 else np.array([])
        elif isinstance(data, np.ndarray):
            return data.flatten()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _run_model_detection(self, name: str, model: Any, data: np.ndarray) -> Dict[str, Any]:
        """Run detection on a single model."""
        try:
            if hasattr(model, 'detect_anomaly'):
                # Advanced model with detect_anomaly method
                result = model.detect_anomaly(data)
                return {
                    'score': result.get('anomaly_score', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'details': result
                }
            elif hasattr(model, 'score'):
                # Basic model with score method
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                score = model.score(data)
                score_val = float(np.mean(score)) if hasattr(score, '__iter__') else float(score)
                return {
                    'score': score_val,
                    'confidence': min(0.9, abs(score_val) * 2),  # Rough confidence estimate
                    'details': {'raw_score': score_val}
                }
            else:
                # Fallback
                return {'score': 0.0, 'confidence': 0.0, 'error': 'No detection method found'}

        except Exception as e:
            return {'score': 0.0, 'confidence': 0.0, 'error': str(e)}

    def _apply_detection_strategy(
        self,
        strategy: DetectionStrategy,
        model_results: Dict[str, Dict[str, Any]],
        drift_detected: bool
    ) -> DetectionResult:
        """Apply the specified detection strategy to combine model results."""

        # Filter valid results
        valid_results = {
            name: result for name, result in model_results.items()
            if 'error' not in result and result['score'] is not None
        }

        if not valid_results:
            return DetectionResult(
                anomaly_score=0.0,
                is_anomaly=False,
                confidence=0.0,
                anomaly_class=AnomalyClass.UNKNOWN,
                explanation={'error': 'No valid model results'},
                timestamp=time.time()
            )

        if strategy == DetectionStrategy.ENSEMBLE_VOTING:
            return self._ensemble_voting(valid_results, drift_detected)
        elif strategy == DetectionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average(valid_results, drift_detected)
        elif strategy == DetectionStrategy.ADAPTIVE_SELECTION:
            return self._adaptive_selection(valid_results, drift_detected)
        elif strategy == DetectionStrategy.CONFIDENCE_BASED:
            return self._confidence_based(valid_results, drift_detected)
        elif strategy == DetectionStrategy.MULTI_SCALE:
            return self._multi_scale(valid_results, drift_detected)
        else:
            # Default to weighted average
            return self._weighted_average(valid_results, drift_detected)

    def _ensemble_voting(self, results: Dict[str, Dict[str, Any]], drift_detected: bool) -> DetectionResult:
        """Ensemble voting strategy."""
        votes = []
        scores = []
        confidences = []

        for name, result in results.items():
            threshold = self.model_configs[name].threshold
            vote = 1 if result['score'] > threshold else 0
            votes.append(vote)
            scores.append(result['score'])
            confidences.append(result['confidence'])

        # Majority voting
        anomaly_votes = sum(votes)
        is_anomaly = anomaly_votes >= self.config.min_models_agreement

        # Combined metrics
        avg_score = np.mean(scores)
        avg_confidence = np.mean(confidences)
        consensus_ratio = anomaly_votes / len(votes)

        # Classify anomaly type
        anomaly_class = self._classify_anomaly_type(avg_score, consensus_ratio, drift_detected)

        return DetectionResult(
            anomaly_score=avg_score,
            is_anomaly=is_anomaly,
            confidence=avg_confidence * consensus_ratio,
            anomaly_class=anomaly_class,
            explanation={
                'strategy': 'ensemble_voting',
                'votes': dict(zip(results.keys(), votes)),
                'consensus_ratio': consensus_ratio,
                'anomaly_votes': anomaly_votes,
                'total_models': len(votes)
            },
            timestamp=time.time(),
            model_contributions=dict(zip(results.keys(), scores))
        )

    def _weighted_average(self, results: Dict[str, Dict[str, Any]], drift_detected: bool) -> DetectionResult:
        """Weighted average strategy."""
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for name, result in results.items():
            weight = self.model_weights[name]
            weighted_score += result['score'] * weight
            weighted_confidence += result['confidence'] * weight
            total_weight += weight

        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0

        is_anomaly = final_score > self.config.consensus_threshold
        anomaly_class = self._classify_anomaly_type(final_score, final_confidence, drift_detected)

        return DetectionResult(
            anomaly_score=final_score,
            is_anomaly=is_anomaly,
            confidence=final_confidence,
            anomaly_class=anomaly_class,
            explanation={
                'strategy': 'weighted_average',
                'weights': dict(self.model_weights),
                'total_weight': total_weight
            },
            timestamp=time.time(),
            model_contributions={name: result['score'] for name, result in results.items()}
        )

    def _adaptive_selection(self, results: Dict[str, Dict[str, Any]], drift_detected: bool) -> DetectionResult:
        """Adaptive model selection based on recent performance."""
        # Select best performing model
        best_model = self._select_best_model(results.keys())

        if best_model and best_model in results:
            result = results[best_model]
            threshold = self.model_configs[best_model].threshold

            return DetectionResult(
                anomaly_score=result['score'],
                is_anomaly=result['score'] > threshold,
                confidence=result['confidence'],
                anomaly_class=self._classify_anomaly_type(result['score'], result['confidence'], drift_detected),
                explanation={
                    'strategy': 'adaptive_selection',
                    'selected_model': best_model,
                    'selection_reason': 'best_recent_performance'
                },
                timestamp=time.time(),
                model_contributions={best_model: result['score']}
            )
        else:
            # Fallback to weighted average
            return self._weighted_average(results, drift_detected)

    def _confidence_based(self, results: Dict[str, Dict[str, Any]], drift_detected: bool) -> DetectionResult:
        """Confidence-based strategy - weight by model confidence."""
        confidence_weights = {}
        total_confidence = 0.0

        for name, result in results.items():
            confidence = result['confidence']
            confidence_weights[name] = confidence
            total_confidence += confidence

        if total_confidence > 0:
            # Normalize confidence weights
            for name in confidence_weights:
                confidence_weights[name] /= total_confidence

            # Weighted average by confidence
            weighted_score = sum(
                results[name]['score'] * confidence_weights[name]
                for name in results
            )

            avg_confidence = total_confidence / len(results)
        else:
            weighted_score = 0.0
            avg_confidence = 0.0

        is_anomaly = weighted_score > self.config.consensus_threshold
        anomaly_class = self._classify_anomaly_type(weighted_score, avg_confidence, drift_detected)

        return DetectionResult(
            anomaly_score=weighted_score,
            is_anomaly=is_anomaly,
            confidence=avg_confidence,
            anomaly_class=anomaly_class,
            explanation={
                'strategy': 'confidence_based',
                'confidence_weights': confidence_weights
            },
            timestamp=time.time(),
            model_contributions={name: result['score'] for name, result in results.items()}
        )

    def _multi_scale(self, results: Dict[str, Dict[str, Any]], drift_detected: bool) -> DetectionResult:
        """Multi-scale strategy combining different approaches."""
        # Combine voting and weighted average
        voting_result = self._ensemble_voting(results, drift_detected)
        weighted_result = self._weighted_average(results, drift_detected)

        # Meta-ensemble
        final_score = (voting_result.anomaly_score + weighted_result.anomaly_score) / 2
        final_confidence = (voting_result.confidence + weighted_result.confidence) / 2

        # Require agreement from both approaches for positive detection
        is_anomaly = voting_result.is_anomaly and weighted_result.is_anomaly

        anomaly_class = self._classify_anomaly_type(final_score, final_confidence, drift_detected)

        return DetectionResult(
            anomaly_score=final_score,
            is_anomaly=is_anomaly,
            confidence=final_confidence,
            anomaly_class=anomaly_class,
            explanation={
                'strategy': 'multi_scale',
                'voting_result': voting_result.explanation,
                'weighted_result': weighted_result.explanation
            },
            timestamp=time.time(),
            model_contributions={name: result['score'] for name, result in results.items()}
        )

    def _classify_anomaly_type(self, score: float, confidence: float, drift_detected: bool) -> AnomalyClass:
        """Classify the type of anomaly detected."""
        if drift_detected:
            return AnomalyClass.DRIFT_ANOMALY
        elif score > 0.8 and confidence > 0.8:
            return AnomalyClass.POINT_ANOMALY
        elif score > 0.6:
            return AnomalyClass.CONTEXTUAL_ANOMALY
        elif score > 0.4:
            return AnomalyClass.COLLECTIVE_ANOMALY
        else:
            return AnomalyClass.UNKNOWN

    def _select_best_model(self, available_models: List[str]) -> Optional[str]:
        """Select the best performing model based on recent performance."""
        if not self.model_performance:
            return None

        best_model = None
        best_score = -1.0

        for model_name in available_models:
            if model_name in self.model_performance:
                recent_scores = list(self.model_performance[model_name])
                if recent_scores:
                    avg_score = np.mean(recent_scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model_name

        return best_model

    def _update_performance_tracking(self, model_results: Dict[str, Dict[str, Any]], final_result: DetectionResult) -> None:
        """Update performance tracking for adaptive weights."""
        for name, result in model_results.items():
            if 'error' not in result:
                # Simple performance metric based on confidence
                performance_score = result['confidence']
                self.model_performance[name].append(performance_score)

        # Update adaptive weights if enabled
        if self.config.enable_adaptive_weights:
            self._update_adaptive_weights()

    def _update_adaptive_weights(self) -> None:
        """Update model weights based on recent performance."""
        for name in self.model_weights:
            if self.model_performance.get(name):
                recent_performance = list(self.model_performance[name])
                if len(recent_performance) >= 10:  # Need enough data points
                    avg_performance = np.mean(recent_performance)
                    # Adjust weight based on performance (simple approach)
                    self.model_weights[name] = max(0.1, min(2.0, avg_performance * 2))

    async def _generate_explanation(
        self,
        data: np.ndarray,
        model_results: Dict[str, Dict[str, Any]],
        final_result: DetectionResult
    ) -> Dict[str, Any]:
        """Generate explanation for the detection result."""
        try:
            # Basic explanation
            explanation = {
                'feature_importance': {},
                'model_agreement': {},
                'decision_factors': []
            }

            # Analyze model agreement
            scores = [result['score'] for result in model_results.values() if 'error' not in result]
            if scores:
                explanation['model_agreement'] = {
                    'score_variance': float(np.var(scores)),
                    'score_range': [float(min(scores)), float(max(scores))],
                    'agreement_level': 'high' if np.var(scores) < 0.1 else 'low'
                }

            # Decision factors
            if final_result.is_anomaly:
                explanation['decision_factors'].append('Anomaly score above threshold')
                if final_result.confidence > 0.8:
                    explanation['decision_factors'].append('High confidence detection')

            return explanation

        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return {'error': str(e)}

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        with self.model_lock:
            stats = {
                'total_detections': len(self.detection_history),
                'model_performance': dict(self.model_performance),
                'current_weights': dict(self.model_weights),
                'enabled_models': [name for name, cfg in self.model_configs.items() if cfg.enabled],
                'recent_anomaly_rate': 0.0,
                'average_confidence': 0.0
            }

            if self.detection_history:
                recent_results = list(self.detection_history)[-100:]  # Last 100 detections
                anomaly_count = sum(1 for r in recent_results if r.is_anomaly)
                stats['recent_anomaly_rate'] = anomaly_count / len(recent_results)
                stats['average_confidence'] = np.mean([r.confidence for r in recent_results])

            return stats

    def update_model_config(self, model_name: str, **kwargs) -> None:
        """Update configuration for a specific model."""
        if model_name in self.model_configs:
            config = self.model_configs[model_name]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    self.logger.info(f"Updated {model_name}.{key} = {value}")

            # Update weight if changed
            if 'weight' in kwargs:
                self.model_weights[model_name] = kwargs['weight']

    def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        self.executor.shutdown(wait=True)
        self.logger.info("Orchestrator shutdown complete")


# CLI Interface
def main() -> None:
    """CLI entry point for intelligent anomaly orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Intelligent Anomaly Detection Orchestrator"
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to orchestrator configuration JSON file"
    )
    parser.add_argument(
        "--input-file",
        help="Input CSV file for batch processing"
    )
    parser.add_argument(
        "--output-file",
        default="orchestrator_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in DetectionStrategy],
        help="Detection strategy to use"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)

    # Load configuration
    with open(args.config_file) as f:
        config_data = json.load(f)

    # Parse model configurations
    models_config = [ModelConfig(**cfg) for cfg in config_data['models']]

    # Parse orchestration config
    orch_config = OrchestrationConfig(**config_data.get('orchestration', {}))
    if args.strategy:
        orch_config.strategy = DetectionStrategy(args.strategy)

    # Create orchestrator
    orchestrator = IntelligentAnomalyOrchestrator(models_config, orch_config)

    async def run_orchestrator():
        if args.input_file:
            # Batch processing mode
            logger.info(f"Processing file: {args.input_file}")
            df = pd.read_csv(args.input_file)

            results = []
            for idx, row in df.iterrows():
                result = await orchestrator.detect_anomaly(row.to_dict())
                results.append({
                    'index': idx,
                    'anomaly_score': result.anomaly_score,
                    'is_anomaly': result.is_anomaly,
                    'confidence': result.confidence,
                    'anomaly_class': result.anomaly_class.value,
                    'timestamp': result.timestamp,
                    'model_contributions': result.model_contributions,
                    'explanation': result.explanation
                })

            # Save results
            with open(args.output_file, 'w') as f:
                json.dump({
                    'results': results,
                    'stats': orchestrator.get_orchestration_stats()
                }, f, indent=2)

            logger.info(f"Results saved to {args.output_file}")
        else:
            # Interactive mode
            logger.info("Starting interactive mode. Enter 'quit' to exit.")
            logger.info("Submit JSON data for anomaly detection.")

            # Example usage - would typically integrate with real-time data streams
            logger.info("Orchestrator ready for real-time processing")

    asyncio.run(run_orchestrator())


if __name__ == "__main__":
    main()
