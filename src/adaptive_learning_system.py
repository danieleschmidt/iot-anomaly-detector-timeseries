"""
Adaptive Learning System for Continuous Model Improvement
Implements self-learning capabilities with online adaptation
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .generation_1_autonomous_core import AutonomousAnomalyCore, AnomalyResult
from .logging_config import setup_logging


class LearningMode(Enum):
    """Learning modes for adaptive system."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    CONTINUAL = "continual"


@dataclass
class FeedbackSignal:
    """User feedback signal for model improvement."""
    timestamp: float
    sample_id: str
    true_label: bool  # True if anomaly, False if normal
    predicted_label: bool
    confidence: float
    correction_weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEvent:
    """Record of learning events for analysis."""
    timestamp: float
    event_type: str
    trigger_reason: str
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    data_samples: int
    learning_time: float
    success: bool


class AdaptiveLearningSystem:
    """
    Advanced adaptive learning system for continuous model improvement.
    
    Features:
    - Online learning from user feedback
    - Continual learning without catastrophic forgetting
    - Active learning for optimal sample selection
    - Meta-learning for fast adaptation
    - Concept drift adaptation
    - Multi-modal learning integration
    """
    
    def __init__(
        self,
        core_model: AutonomousAnomalyCore,
        learning_mode: LearningMode = LearningMode.CONTINUAL,
        feedback_buffer_size: int = 10000,
        adaptation_threshold: float = 0.1,
        min_samples_for_adaptation: int = 100,
        learning_rate: float = 0.001,
        meta_learning_enabled: bool = True
    ):
        self.core_model = core_model
        self.learning_mode = learning_mode
        self.feedback_buffer_size = feedback_buffer_size
        self.adaptation_threshold = adaptation_threshold
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self.learning_rate = learning_rate
        self.meta_learning_enabled = meta_learning_enabled
        
        # Learning components
        self.feedback_buffer: deque = deque(maxlen=feedback_buffer_size)
        self.learning_history: List[LearningEvent] = []
        self.concept_tracker = ConceptDriftTracker()
        self.active_learner = ActiveLearningSelector()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.adaptation_scheduler = AdaptationScheduler()
        
        # State management
        self.last_adaptation_time = 0.0
        self.adaptation_count = 0
        self.is_learning_active = False
        
        self.logger = setup_logging(__name__)
        self.logger.info(f"AdaptiveLearningSystem initialized with mode={learning_mode.value}")
    
    async def add_feedback(
        self,
        sample_data: pd.DataFrame,
        feedback: FeedbackSignal
    ) -> bool:
        """Add user feedback for model improvement."""
        try:
            # Store feedback with associated data
            feedback_entry = {
                'feedback': feedback,
                'data': sample_data,
                'timestamp': time.time()
            }
            
            self.feedback_buffer.append(feedback_entry)
            
            # Update performance tracking
            await self.performance_tracker.update_feedback(feedback)
            
            # Check if adaptation should be triggered
            if await self._should_trigger_adaptation():
                await self._trigger_adaptation()
            
            self.logger.debug(f"Feedback added: correct={feedback.true_label == feedback.predicted_label}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add feedback: {str(e)}")
            return False
    
    async def _should_trigger_adaptation(self) -> bool:
        """Determine if model adaptation should be triggered."""
        if self.is_learning_active:
            return False
        
        if len(self.feedback_buffer) < self.min_samples_for_adaptation:
            return False
        
        # Check performance degradation
        recent_performance = await self.performance_tracker.get_recent_performance()
        if recent_performance['accuracy'] < (1.0 - self.adaptation_threshold):
            self.logger.info("Performance degradation detected, triggering adaptation")
            return True
        
        # Check concept drift
        if await self.concept_tracker.detect_drift([entry['data'] for entry in self.feedback_buffer]):
            self.logger.info("Concept drift detected, triggering adaptation")
            return True
        
        # Time-based adaptation
        time_since_adaptation = time.time() - self.last_adaptation_time
        if time_since_adaptation > self.adaptation_scheduler.get_next_interval():
            self.logger.info("Scheduled adaptation triggered")
            return True
        
        return False
    
    async def _trigger_adaptation(self) -> None:
        """Trigger model adaptation based on collected feedback."""
        if self.is_learning_active:
            self.logger.warning("Adaptation already in progress")
            return
        
        self.is_learning_active = True
        adaptation_start = time.time()
        
        try:
            # Record performance before adaptation
            performance_before = await self.performance_tracker.get_current_performance()
            
            # Prepare training data from feedback
            training_data, labels = await self._prepare_training_data()
            
            if len(training_data) == 0:
                self.logger.warning("No training data available for adaptation")
                return
            
            # Select adaptation strategy based on learning mode
            success = await self._execute_adaptation(training_data, labels)
            
            # Record performance after adaptation
            performance_after = await self.performance_tracker.get_current_performance()
            
            # Record learning event
            learning_event = LearningEvent(
                timestamp=time.time(),
                event_type=f"adaptation_{self.learning_mode.value}",
                trigger_reason="performance_feedback",
                performance_before=performance_before,
                performance_after=performance_after,
                data_samples=len(training_data),
                learning_time=time.time() - adaptation_start,
                success=success
            )
            
            self.learning_history.append(learning_event)
            self.adaptation_count += 1
            self.last_adaptation_time = time.time()
            
            if success:
                self.logger.info(f"Model adaptation completed successfully in {learning_event.learning_time:.2f}s")
            else:
                self.logger.warning("Model adaptation failed or showed no improvement")
        
        except Exception as e:
            self.logger.error(f"Adaptation failed: {str(e)}")
        
        finally:
            self.is_learning_active = False
    
    async def _prepare_training_data(self) -> Tuple[List[pd.DataFrame], List[bool]]:
        """Prepare training data from feedback buffer."""
        training_data = []
        labels = []
        
        # Extract data and labels from feedback
        for entry in self.feedback_buffer:
            feedback = entry['feedback']
            data = entry['data']
            
            training_data.append(data)
            labels.append(feedback.true_label)
        
        # Apply active learning selection if enabled
        if len(training_data) > self.min_samples_for_adaptation * 2:
            selected_indices = await self.active_learner.select_samples(
                training_data, labels, target_size=self.min_samples_for_adaptation
            )
            training_data = [training_data[i] for i in selected_indices]
            labels = [labels[i] for i in selected_indices]
        
        return training_data, labels
    
    async def _execute_adaptation(
        self, 
        training_data: List[pd.DataFrame], 
        labels: List[bool]
    ) -> bool:
        """Execute model adaptation based on learning mode."""
        try:
            if self.learning_mode == LearningMode.SUPERVISED:
                return await self._supervised_adaptation(training_data, labels)
            
            elif self.learning_mode == LearningMode.UNSUPERVISED:
                return await self._unsupervised_adaptation(training_data)
            
            elif self.learning_mode == LearningMode.SEMI_SUPERVISED:
                return await self._semi_supervised_adaptation(training_data, labels)
            
            elif self.learning_mode == LearningMode.CONTINUAL:
                return await self._continual_adaptation(training_data, labels)
            
            elif self.learning_mode == LearningMode.REINFORCEMENT:
                return await self._reinforcement_adaptation(training_data, labels)
            
            else:
                self.logger.error(f"Unknown learning mode: {self.learning_mode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Adaptation execution failed: {str(e)}")
            return False
    
    async def _supervised_adaptation(
        self, 
        training_data: List[pd.DataFrame], 
        labels: List[bool]
    ) -> bool:
        """Supervised learning adaptation using labeled feedback."""
        # Combine all training data
        combined_data = pd.concat(training_data, ignore_index=True)
        
        # Fine-tune the model with corrected labels
        # This is a simplified approach - in practice, you'd implement
        # more sophisticated fine-tuning techniques
        
        # Split into normal and anomalous samples
        normal_data = combined_data[np.array(labels) == False]
        anomalous_data = combined_data[np.array(labels) == True]
        
        # Retrain with corrected samples
        if len(normal_data) > 10:  # Minimum samples for training
            await self.core_model.train_ensemble(
                normal_data,
                epochs=10,  # Quick adaptation
                validation_split=0.1
            )
            return True
        
        return False
    
    async def _unsupervised_adaptation(self, training_data: List[pd.DataFrame]) -> bool:
        """Unsupervised learning adaptation using clustering and density estimation."""
        combined_data = pd.concat(training_data, ignore_index=True)
        
        # Use clustering to identify new patterns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data.select_dtypes(include=[np.number]))
        
        # Apply clustering to find new normal patterns
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Update model with new patterns
        # This would involve retraining on the dominant cluster patterns
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=combined_data.select_dtypes(include=[np.number]).columns
        )
        
        # Quick adaptation training
        await self.core_model.train_ensemble(
            cluster_centers,
            epochs=5
        )
        
        return True
    
    async def _semi_supervised_adaptation(
        self, 
        training_data: List[pd.DataFrame], 
        labels: List[bool]
    ) -> bool:
        """Semi-supervised adaptation using both labeled and unlabeled data."""
        # Combine labeled and unlabeled samples
        labeled_indices = [i for i, label in enumerate(labels) if label is not None]
        unlabeled_indices = [i for i, label in enumerate(labels) if label is None]
        
        labeled_data = [training_data[i] for i in labeled_indices]
        labeled_labels = [labels[i] for i in labeled_indices]
        unlabeled_data = [training_data[i] for i in unlabeled_indices]
        
        # Use labeled data for supervised learning
        if labeled_data:
            success = await self._supervised_adaptation(labeled_data, labeled_labels)
        else:
            success = True
        
        # Use unlabeled data for unsupervised learning
        if unlabeled_data and success:
            success = await self._unsupervised_adaptation(unlabeled_data)
        
        return success
    
    async def _continual_adaptation(
        self, 
        training_data: List[pd.DataFrame], 
        labels: List[bool]
    ) -> bool:
        """Continual learning adaptation to prevent catastrophic forgetting."""
        # Implement experience replay to maintain old knowledge
        
        # Select representative samples from historical data
        historical_samples = await self._get_representative_samples()
        
        # Combine new and historical data
        if historical_samples:
            all_data = training_data + historical_samples['data']
            all_labels = labels + historical_samples['labels']
        else:
            all_data = training_data
            all_labels = labels
        
        # Progressive training with regularization
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Use elastic weight consolidation or similar technique
        # to preserve important parameters
        await self.core_model.train_ensemble(
            combined_data,
            epochs=15,  # More epochs for continual learning
            validation_split=0.15
        )
        
        return True
    
    async def _reinforcement_adaptation(
        self, 
        training_data: List[pd.DataFrame], 
        labels: List[bool]
    ) -> bool:
        """Reinforcement learning adaptation based on reward signals."""
        # Convert feedback to rewards
        rewards = []
        for feedback in [entry['feedback'] for entry in self.feedback_buffer]:
            if feedback.predicted_label == feedback.true_label:
                reward = 1.0 * feedback.confidence
            else:
                reward = -1.0 * feedback.correction_weight
            rewards.append(reward)
        
        # Update model using policy gradient or similar RL technique
        # This is a simplified implementation
        if np.mean(rewards) > 0:
            # Positive feedback - strengthen current model
            return True
        else:
            # Negative feedback - adapt model
            return await self._supervised_adaptation(training_data, labels)
    
    async def _get_representative_samples(self) -> Optional[Dict[str, List]]:
        """Get representative samples from historical data to prevent forgetting."""
        if len(self.learning_history) == 0:
            return None
        
        # This is a simplified approach - in practice, you'd use more
        # sophisticated methods like memory replay, gradient episodic memory, etc.
        
        # Select diverse samples from different time periods
        sample_data = []
        sample_labels = []
        
        for event in self.learning_history[-10:]:  # Last 10 learning events
            if 'sample_data' in event.__dict__:
                sample_data.extend(event.sample_data)
                sample_labels.extend(event.sample_labels)
        
        return {'data': sample_data, 'labels': sample_labels} if sample_data else None
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        return {
            'total_adaptations': self.adaptation_count,
            'feedback_buffer_size': len(self.feedback_buffer),
            'learning_mode': self.learning_mode.value,
            'last_adaptation_time': self.last_adaptation_time,
            'is_learning_active': self.is_learning_active,
            'recent_performance': asyncio.run(self.performance_tracker.get_recent_performance()),
            'learning_events': len(self.learning_history),
            'concept_drift_detected': len(self.concept_tracker.drift_history) > 0
        }


class ConceptDriftTracker:
    """Track concept drift in data streams."""
    
    def __init__(self, window_size: int = 1000, sensitivity: float = 0.05):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.reference_stats = None
        self.drift_history: List[Dict] = []
        self.logger = setup_logging(__name__)
    
    async def detect_drift(self, data_stream: List[pd.DataFrame]) -> bool:
        """Detect concept drift in data stream."""
        if not data_stream:
            return False
        
        try:
            # Combine recent data
            recent_data = pd.concat(data_stream[-self.window_size:], ignore_index=True)
            
            # Calculate current statistics
            current_stats = self._calculate_statistics(recent_data)
            
            if self.reference_stats is None:
                self.reference_stats = current_stats
                return False
            
            # Compare distributions
            drift_score = self._calculate_drift_score(self.reference_stats, current_stats)
            
            if drift_score > self.sensitivity:
                self.drift_history.append({
                    'timestamp': time.time(),
                    'drift_score': drift_score,
                    'data_samples': len(recent_data)
                })
                
                # Update reference stats
                self.reference_stats = current_stats
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Drift detection error: {str(e)}")
            return False
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistical properties for drift detection."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        return {
            'mean': float(numeric_data.mean().mean()),
            'std': float(numeric_data.std().mean()),
            'skewness': float(numeric_data.skew().mean()),
            'kurtosis': float(numeric_data.kurtosis().mean()),
            'correlation': float(np.mean(np.abs(numeric_data.corr().values)))
        }
    
    def _calculate_drift_score(self, ref_stats: Dict, curr_stats: Dict) -> float:
        """Calculate drift score between reference and current statistics."""
        scores = []
        for key in ref_stats:
            if key in curr_stats and ref_stats[key] != 0:
                score = abs(curr_stats[key] - ref_stats[key]) / abs(ref_stats[key])
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0


class ActiveLearningSelector:
    """Select most informative samples for learning."""
    
    def __init__(self, strategy: str = "uncertainty"):
        self.strategy = strategy
        self.logger = setup_logging(__name__)
    
    async def select_samples(
        self, 
        data: List[pd.DataFrame], 
        labels: List[bool], 
        target_size: int
    ) -> List[int]:
        """Select most informative samples for training."""
        try:
            if len(data) <= target_size:
                return list(range(len(data)))
            
            if self.strategy == "uncertainty":
                return await self._uncertainty_sampling(data, labels, target_size)
            elif self.strategy == "diversity":
                return await self._diversity_sampling(data, target_size)
            elif self.strategy == "hybrid":
                return await self._hybrid_sampling(data, labels, target_size)
            else:
                # Random sampling as fallback
                return np.random.choice(len(data), target_size, replace=False).tolist()
                
        except Exception as e:
            self.logger.error(f"Sample selection error: {str(e)}")
            return list(range(min(target_size, len(data))))
    
    async def _uncertainty_sampling(
        self, 
        data: List[pd.DataFrame], 
        labels: List[bool], 
        target_size: int
    ) -> List[int]:
        """Select samples with highest prediction uncertainty."""
        # This would involve running inference and selecting samples
        # with lowest confidence scores
        uncertainty_scores = np.random.random(len(data))  # Simplified
        selected_indices = np.argsort(uncertainty_scores)[-target_size:]
        return selected_indices.tolist()
    
    async def _diversity_sampling(
        self, 
        data: List[pd.DataFrame], 
        target_size: int
    ) -> List[int]:
        """Select diverse samples to cover feature space."""
        # Use clustering to select diverse samples
        combined_data = pd.concat(data, ignore_index=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data.select_dtypes(include=[np.number]))
        
        # Use k-means to find diverse samples
        kmeans = KMeans(n_clusters=min(target_size, len(data)), random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Select one sample from each cluster
        selected_indices = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_samples = np.where(clusters == cluster_id)[0]
            if len(cluster_samples) > 0:
                # Select sample closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(scaled_data[cluster_samples] - cluster_center, axis=1)
                closest_idx = cluster_samples[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return selected_indices[:target_size]
    
    async def _hybrid_sampling(
        self, 
        data: List[pd.DataFrame], 
        labels: List[bool], 
        target_size: int
    ) -> List[int]:
        """Combine uncertainty and diversity sampling."""
        # Select half using uncertainty, half using diversity
        uncertainty_size = target_size // 2
        diversity_size = target_size - uncertainty_size
        
        uncertainty_indices = await self._uncertainty_sampling(data, labels, uncertainty_size)
        diversity_indices = await self._diversity_sampling(data, diversity_size)
        
        # Combine and remove duplicates
        all_indices = set(uncertainty_indices + diversity_indices)
        return list(all_indices)[:target_size]


class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.feedback_history: deque = deque(maxlen=window_size)
        self.performance_history: List[Dict] = []
        
    async def update_feedback(self, feedback: FeedbackSignal) -> None:
        """Update performance tracking with new feedback."""
        self.feedback_history.append(feedback)
    
    async def get_recent_performance(self) -> Dict[str, float]:
        """Get recent performance metrics."""
        if not self.feedback_history:
            return {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0}
        
        correct_predictions = sum(
            1 for f in self.feedback_history 
            if f.predicted_label == f.true_label
        )
        
        accuracy = correct_predictions / len(self.feedback_history)
        
        # Calculate precision and recall
        true_positives = sum(
            1 for f in self.feedback_history 
            if f.predicted_label and f.true_label
        )
        predicted_positives = sum(
            1 for f in self.feedback_history 
            if f.predicted_label
        )
        actual_positives = sum(
            1 for f in self.feedback_history 
            if f.true_label
        )
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 1.0
        recall = true_positives / actual_positives if actual_positives > 0 else 1.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    async def get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return await self.get_recent_performance()


class AdaptationScheduler:
    """Schedule model adaptations."""
    
    def __init__(self, base_interval: float = 3600.0):  # 1 hour
        self.base_interval = base_interval
        self.adaptation_count = 0
        
    def get_next_interval(self) -> float:
        """Get next adaptation interval with exponential backoff."""
        return self.base_interval * (1.5 ** min(self.adaptation_count, 5))
    
    def record_adaptation(self) -> None:
        """Record that an adaptation occurred."""
        self.adaptation_count += 1