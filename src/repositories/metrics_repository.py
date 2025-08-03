"""
Metrics Repository

Repository for system and application metrics persistence.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .base_repository import FileBasedRepository, RepositoryException

logger = logging.getLogger(__name__)


class MetricEntity:
    """Entity representing a metric."""
    
    def __init__(
        self,
        metric_id: str,
        name: str,
        value: float,
        metric_type: str,
        timestamp: datetime,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.metric_id = metric_id
        self.name = name
        self.value = value
        self.metric_type = metric_type
        self.timestamp = timestamp
        self.labels = labels or {}
        self.metadata = metadata or {}


class MetricsRepository(FileBasedRepository[MetricEntity]):
    """Repository for metrics storage and retrieval."""
    
    def __init__(self, base_path: str = "metrics"):
        super().__init__(base_path=base_path, file_format='json')
        self._metrics_cache = defaultdict(list)
        self._aggregations = {}
        
    def create(self, entity: MetricEntity) -> MetricEntity:
        """Create a new metric entry."""
        if not entity.metric_id:
            entity.metric_id = f"{entity.name}_{entity.timestamp.isoformat()}"
        
        self._metrics_cache[entity.name].append(entity)
        self._data[entity.metric_id] = entity
        
        # Trigger aggregation if needed
        self._update_aggregations(entity.name)
        
        return entity
    
    def read(self, entity_id: Union[str, int]) -> Optional[MetricEntity]:
        """Read a metric by ID."""
        return self._data.get(entity_id)
    
    def update(self, entity: MetricEntity) -> MetricEntity:
        """Update a metric entry."""
        if entity.metric_id not in self._data:
            raise RepositoryException(f"Metric {entity.metric_id} not found")
        
        self._data[entity.metric_id] = entity
        self._update_aggregations(entity.name)
        
        return entity
    
    def delete(self, entity_id: Union[str, int]) -> bool:
        """Delete a metric entry."""
        if entity_id in self._data:
            entity = self._data[entity_id]
            del self._data[entity_id]
            
            # Remove from cache
            self._metrics_cache[entity.name] = [
                m for m in self._metrics_cache[entity.name]
                if m.metric_id != entity_id
            ]
            
            return True
        return False
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[MetricEntity]:
        """Find all metrics."""
        metrics = list(self._data.values())
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        return self._apply_pagination(metrics, limit, offset)
    
    def find_by(self, criteria: Dict[str, Any]) -> List[MetricEntity]:
        """Find metrics by criteria."""
        results = []
        
        for entity in self._data.values():
            if self._match_metric_criteria(entity, criteria):
                results.append(entity)
        
        return results
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count metrics."""
        if criteria:
            return len(self.find_by(criteria))
        return len(self._data)
    
    def exists(self, entity_id: Union[str, int]) -> bool:
        """Check if metric exists."""
        return entity_id in self._data
    
    def get_metric_series(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricEntity]:
        """Get time series for a metric."""
        metrics = self._metrics_cache.get(name, [])
        
        # Apply filters
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        if labels:
            metrics = [m for m in metrics if self._match_labels(m.labels, labels)]
        
        return sorted(metrics, key=lambda x: x.timestamp)
    
    def get_aggregated_metrics(
        self,
        name: str,
        aggregation: str,
        window: str = '5m'
    ) -> Dict[str, Any]:
        """Get aggregated metrics."""
        key = f"{name}_{aggregation}_{window}"
        
        if key in self._aggregations:
            return self._aggregations[key]
        
        # Calculate aggregation
        metrics = self._metrics_cache.get(name, [])
        if not metrics:
            return {}
        
        # Parse window
        window_seconds = self._parse_window(window)
        
        # Group by window
        windows = defaultdict(list)
        for metric in metrics:
            window_key = int(metric.timestamp.timestamp() / window_seconds)
            windows[window_key].append(metric.value)
        
        # Apply aggregation
        result = {}
        for window_key, values in windows.items():
            timestamp = datetime.fromtimestamp(window_key * window_seconds)
            
            if aggregation == 'avg':
                result[timestamp.isoformat()] = sum(values) / len(values)
            elif aggregation == 'sum':
                result[timestamp.isoformat()] = sum(values)
            elif aggregation == 'max':
                result[timestamp.isoformat()] = max(values)
            elif aggregation == 'min':
                result[timestamp.isoformat()] = min(values)
            elif aggregation == 'count':
                result[timestamp.isoformat()] = len(values)
        
        self._aggregations[key] = result
        return result
    
    def record_counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> MetricEntity:
        """Record a counter metric."""
        entity = MetricEntity(
            metric_id=f"{name}_{datetime.now().timestamp()}",
            name=name,
            value=value,
            metric_type='counter',
            timestamp=datetime.now(),
            labels=labels
        )
        return self.create(entity)
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> MetricEntity:
        """Record a gauge metric."""
        entity = MetricEntity(
            metric_id=f"{name}_{datetime.now().timestamp()}",
            name=name,
            value=value,
            metric_type='gauge',
            timestamp=datetime.now(),
            labels=labels
        )
        return self.create(entity)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> MetricEntity:
        """Record a histogram metric."""
        metadata = {'buckets': buckets} if buckets else {}
        
        entity = MetricEntity(
            metric_id=f"{name}_{datetime.now().timestamp()}",
            name=name,
            value=value,
            metric_type='histogram',
            timestamp=datetime.now(),
            labels=labels,
            metadata=metadata
        )
        return self.create(entity)
    
    def cleanup_old_metrics(self, retention_hours: int = 24) -> int:
        """Clean up old metrics."""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        to_delete = []
        for metric_id, entity in self._data.items():
            if entity.timestamp < cutoff_time:
                to_delete.append(metric_id)
        
        for metric_id in to_delete:
            self.delete(metric_id)
        
        return len(to_delete)
    
    def _match_metric_criteria(
        self,
        entity: MetricEntity,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if metric matches criteria."""
        for key, value in criteria.items():
            if key == 'name' and entity.name != value:
                return False
            elif key == 'metric_type' and entity.metric_type != value:
                return False
            elif key == 'start_time' and entity.timestamp < value:
                return False
            elif key == 'end_time' and entity.timestamp > value:
                return False
            elif key == 'labels':
                if not self._match_labels(entity.labels, value):
                    return False
        
        return True
    
    def _match_labels(
        self,
        entity_labels: Dict[str, str],
        required_labels: Dict[str, str]
    ) -> bool:
        """Check if labels match."""
        for key, value in required_labels.items():
            if key not in entity_labels or entity_labels[key] != value:
                return False
        return True
    
    def _update_aggregations(self, metric_name: str):
        """Update aggregations for a metric."""
        # Clear cached aggregations for this metric
        keys_to_clear = [
            k for k in self._aggregations.keys()
            if k.startswith(f"{metric_name}_")
        ]
        for key in keys_to_clear:
            del self._aggregations[key]
    
    def _parse_window(self, window: str) -> int:
        """Parse window string to seconds."""
        if window.endswith('s'):
            return int(window[:-1])
        elif window.endswith('m'):
            return int(window[:-1]) * 60
        elif window.endswith('h'):
            return int(window[:-1]) * 3600
        elif window.endswith('d'):
            return int(window[:-1]) * 86400
        else:
            return 300  # Default 5 minutes
    
    def _create_connection(self):
        """Create connection."""
        return True
    
    def _close_connection(self):
        """Close connection."""
        self._save_data()