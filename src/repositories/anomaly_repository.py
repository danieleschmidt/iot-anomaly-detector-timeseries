"""
Anomaly Repository

Repository for anomaly detection results and patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import hashlib

from .base_repository import FileBasedRepository, RepositoryException

logger = logging.getLogger(__name__)


class AnomalyEntity:
    """Entity representing a detected anomaly."""
    
    def __init__(
        self,
        anomaly_id: str,
        sensor_id: str,
        timestamp: datetime,
        severity: str,
        reconstruction_error: float,
        threshold: float,
        model_version: str,
        feature_contributions: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.anomaly_id = anomaly_id
        self.sensor_id = sensor_id
        self.timestamp = timestamp
        self.severity = severity
        self.reconstruction_error = reconstruction_error
        self.threshold = threshold
        self.model_version = model_version
        self.feature_contributions = feature_contributions or {}
        self.metadata = metadata or {}
        self.acknowledged = False
        self.resolved = False


class AnomalyRepository(FileBasedRepository[AnomalyEntity]):
    """Repository for anomaly data persistence and retrieval."""
    
    def __init__(self, base_path: str = "data/anomalies"):
        super().__init__(base_path=base_path, file_format='json')
        self._patterns = {}
        self._severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
    def create(self, entity: AnomalyEntity) -> AnomalyEntity:
        """Create a new anomaly record."""
        if not entity.anomaly_id:
            entity.anomaly_id = self._generate_anomaly_id(entity)
        
        # Check for duplicate
        if self.exists(entity.anomaly_id):
            raise RepositoryException(f"Anomaly {entity.anomaly_id} already exists")
        
        # Update severity counts
        self._severity_counts[entity.severity] += 1
        
        # Detect patterns
        self._detect_pattern(entity)
        
        # Store anomaly
        self._data[entity.anomaly_id] = entity
        self._save_entity(entity.anomaly_id, entity.__dict__)
        
        logger.info(f"Created anomaly: {entity.anomaly_id}")
        return entity
    
    def read(self, entity_id: Union[str, int]) -> Optional[AnomalyEntity]:
        """Read an anomaly by ID."""
        if entity_id in self._data:
            return self._data[entity_id]
        
        # Try to load from file
        entity_file = self.base_path / f"{entity_id}.json"
        if entity_file.exists():
            with open(entity_file, 'r') as f:
                data = json.load(f)
            entity = AnomalyEntity(**data)
            self._data[entity_id] = entity
            return entity
        
        return None
    
    def update(self, entity: AnomalyEntity) -> AnomalyEntity:
        """Update an anomaly record."""
        if not self.exists(entity.anomaly_id):
            raise RepositoryException(f"Anomaly {entity.anomaly_id} not found")
        
        # Update entity
        self._data[entity.anomaly_id] = entity
        self._save_entity(entity.anomaly_id, entity.__dict__)
        
        logger.info(f"Updated anomaly: {entity.anomaly_id}")
        return entity
    
    def delete(self, entity_id: Union[str, int]) -> bool:
        """Delete an anomaly record."""
        if entity_id in self._data:
            entity = self._data[entity_id]
            self._severity_counts[entity.severity] -= 1
            del self._data[entity_id]
            
            # Delete file
            entity_file = self.base_path / f"{entity_id}.json"
            if entity_file.exists():
                entity_file.unlink()
            
            return True
        return False
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[AnomalyEntity]:
        """Find all anomalies."""
        # Load all anomaly files
        for anomaly_file in self.base_path.glob("*.json"):
            anomaly_id = anomaly_file.stem
            if anomaly_id not in self._data:
                self.read(anomaly_id)
        
        anomalies = list(self._data.values())
        anomalies.sort(key=lambda x: x.timestamp, reverse=True)
        
        return self._apply_pagination(anomalies, limit, offset)
    
    def find_by(self, criteria: Dict[str, Any]) -> List[AnomalyEntity]:
        """Find anomalies by criteria."""
        # Load all anomalies first
        self.find_all()
        
        results = []
        for entity in self._data.values():
            if self._match_anomaly_criteria(entity, criteria):
                results.append(entity)
        
        return results
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count anomalies."""
        if criteria:
            return len(self.find_by(criteria))
        
        # Count files if not all loaded
        return len(list(self.base_path.glob("*.json")))
    
    def exists(self, entity_id: Union[str, int]) -> bool:
        """Check if anomaly exists."""
        if entity_id in self._data:
            return True
        
        entity_file = self.base_path / f"{entity_id}.json"
        return entity_file.exists()
    
    def find_by_sensor(
        self,
        sensor_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AnomalyEntity]:
        """Find anomalies for a specific sensor."""
        criteria = {'sensor_id': sensor_id}
        
        if start_time:
            criteria['start_time'] = start_time
        if end_time:
            criteria['end_time'] = end_time
        
        return self.find_by(criteria)
    
    def find_by_severity(
        self,
        severity: str,
        limit: Optional[int] = None
    ) -> List[AnomalyEntity]:
        """Find anomalies by severity level."""
        anomalies = self.find_by({'severity': severity})
        
        if limit:
            anomalies = anomalies[:limit]
        
        return anomalies
    
    def find_unresolved(self) -> List[AnomalyEntity]:
        """Find unresolved anomalies."""
        return self.find_by({'resolved': False})
    
    def find_patterns(
        self,
        min_occurrences: int = 3
    ) -> Dict[str, List[AnomalyEntity]]:
        """Find recurring anomaly patterns."""
        patterns = {}
        
        for pattern_id, anomaly_ids in self._patterns.items():
            if len(anomaly_ids) >= min_occurrences:
                anomalies = [self.read(aid) for aid in anomaly_ids]
                patterns[pattern_id] = [a for a in anomalies if a]
        
        return patterns
    
    def acknowledge_anomaly(
        self,
        anomaly_id: str,
        user: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Acknowledge an anomaly."""
        entity = self.read(anomaly_id)
        if not entity:
            return False
        
        entity.acknowledged = True
        entity.metadata['acknowledged_at'] = datetime.now().isoformat()
        entity.metadata['acknowledged_by'] = user
        entity.metadata['acknowledgment_notes'] = notes
        
        self.update(entity)
        return True
    
    def resolve_anomaly(
        self,
        anomaly_id: str,
        resolution: str,
        user: Optional[str] = None
    ) -> bool:
        """Mark anomaly as resolved."""
        entity = self.read(anomaly_id)
        if not entity:
            return False
        
        entity.resolved = True
        entity.metadata['resolved_at'] = datetime.now().isoformat()
        entity.metadata['resolved_by'] = user
        entity.metadata['resolution'] = resolution
        
        self.update(entity)
        return True
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get anomaly statistics."""
        criteria = {}
        if start_time:
            criteria['start_time'] = start_time
        if end_time:
            criteria['end_time'] = end_time
        
        anomalies = self.find_by(criteria)
        
        if not anomalies:
            return {
                'total': 0,
                'by_severity': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
                'resolved_rate': 0,
                'acknowledged_rate': 0
            }
        
        # Calculate statistics
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        resolved_count = 0
        acknowledged_count = 0
        
        for anomaly in anomalies:
            severity_counts[anomaly.severity] += 1
            if anomaly.resolved:
                resolved_count += 1
            if anomaly.acknowledged:
                acknowledged_count += 1
        
        # Calculate average reconstruction error by severity
        severity_errors = defaultdict(list)
        for anomaly in anomalies:
            severity_errors[anomaly.severity].append(anomaly.reconstruction_error)
        
        avg_errors = {}
        for severity, errors in severity_errors.items():
            avg_errors[severity] = sum(errors) / len(errors) if errors else 0
        
        return {
            'total': len(anomalies),
            'by_severity': severity_counts,
            'resolved_rate': resolved_count / len(anomalies) * 100,
            'acknowledged_rate': acknowledged_count / len(anomalies) * 100,
            'avg_reconstruction_error': avg_errors,
            'patterns_detected': len(self._patterns),
            'time_range': {
                'start': min(a.timestamp for a in anomalies).isoformat(),
                'end': max(a.timestamp for a in anomalies).isoformat()
            }
        }
    
    def get_trending_sensors(
        self,
        hours: int = 24,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get sensors with most anomalies."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        anomalies = self.find_by({'start_time': cutoff_time})
        
        # Count by sensor
        sensor_counts = defaultdict(int)
        for anomaly in anomalies:
            sensor_counts[anomaly.sensor_id] += 1
        
        # Sort and limit
        trending = sorted(
            sensor_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [
            {'sensor_id': sensor_id, 'anomaly_count': count}
            for sensor_id, count in trending
        ]
    
    def _generate_anomaly_id(self, entity: AnomalyEntity) -> str:
        """Generate unique anomaly ID."""
        hash_input = f"{entity.sensor_id}_{entity.timestamp}_{entity.reconstruction_error}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _detect_pattern(self, entity: AnomalyEntity):
        """Detect if anomaly is part of a pattern."""
        # Simple pattern detection based on sensor and severity
        pattern_key = f"{entity.sensor_id}_{entity.severity}"
        
        if pattern_key not in self._patterns:
            self._patterns[pattern_key] = []
        
        self._patterns[pattern_key].append(entity.anomaly_id)
    
    def _match_anomaly_criteria(
        self,
        entity: AnomalyEntity,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if anomaly matches criteria."""
        for key, value in criteria.items():
            if key == 'sensor_id' and entity.sensor_id != value:
                return False
            elif key == 'severity' and entity.severity != value:
                return False
            elif key == 'resolved' and entity.resolved != value:
                return False
            elif key == 'acknowledged' and entity.acknowledged != value:
                return False
            elif key == 'model_version' and entity.model_version != value:
                return False
            elif key == 'start_time' and entity.timestamp < value:
                return False
            elif key == 'end_time' and entity.timestamp > value:
                return False
        
        return True
    
    def _create_connection(self):
        """Create connection."""
        return True
    
    def _close_connection(self):
        """Close connection."""
        self._save_data()

from collections import defaultdict