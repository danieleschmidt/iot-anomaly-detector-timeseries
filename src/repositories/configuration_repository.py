"""
Configuration Repository

Repository for system configuration management.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

from .base_repository import FileBasedRepository, RepositoryException

logger = logging.getLogger(__name__)


class ConfigurationEntity:
    """Entity representing a configuration item."""
    
    def __init__(
        self,
        config_id: str,
        namespace: str,
        key: str,
        value: Any,
        config_type: str = 'string',
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None
    ):
        self.config_id = config_id
        self.namespace = namespace
        self.key = key
        self.value = value
        self.config_type = config_type
        self.description = description
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()


class ConfigurationRepository(FileBasedRepository[ConfigurationEntity]):
    """Repository for configuration persistence and retrieval."""
    
    def __init__(self, base_path: str = "config/repository"):
        super().__init__(base_path=base_path, file_format='json')
        self._config_cache = {}
        self._namespace_index = defaultdict(list)
        self._load_default_configs()
        
    def create(self, entity: ConfigurationEntity) -> ConfigurationEntity:
        """Create a new configuration."""
        if not entity.config_id:
            entity.config_id = f"{entity.namespace}.{entity.key}"
        
        if self.exists(entity.config_id):
            raise RepositoryException(f"Configuration {entity.config_id} already exists")
        
        # Validate value type
        self._validate_config_type(entity.value, entity.config_type)
        
        # Add to namespace index
        self._namespace_index[entity.namespace].append(entity.config_id)
        
        # Store configuration
        entity.created_at = datetime.now().isoformat()
        entity.updated_at = entity.created_at
        
        self._data[entity.config_id] = entity
        self._config_cache[entity.config_id] = entity.value
        self._save_entity(entity.config_id, entity.__dict__)
        
        logger.info(f"Created configuration: {entity.config_id}")
        return entity
    
    def read(self, entity_id: Union[str, int]) -> Optional[ConfigurationEntity]:
        """Read a configuration by ID."""
        return self._data.get(entity_id)
    
    def update(self, entity: ConfigurationEntity) -> ConfigurationEntity:
        """Update a configuration."""
        if not self.exists(entity.config_id):
            raise RepositoryException(f"Configuration {entity.config_id} not found")
        
        # Validate value type
        self._validate_config_type(entity.value, entity.config_type)
        
        # Keep original creation time
        existing = self._data[entity.config_id]
        entity.created_at = existing.created_at
        entity.updated_at = datetime.now().isoformat()
        
        # Update configuration
        self._data[entity.config_id] = entity
        self._config_cache[entity.config_id] = entity.value
        self._save_entity(entity.config_id, entity.__dict__)
        
        logger.info(f"Updated configuration: {entity.config_id}")
        return entity
    
    def delete(self, entity_id: Union[str, int]) -> bool:
        """Delete a configuration."""
        if entity_id in self._data:
            entity = self._data[entity_id]
            
            # Remove from namespace index
            self._namespace_index[entity.namespace].remove(entity_id)
            
            # Delete configuration
            del self._data[entity_id]
            if entity_id in self._config_cache:
                del self._config_cache[entity_id]
            
            self._delete_entity_file(entity_id)
            
            logger.info(f"Deleted configuration: {entity_id}")
            return True
        return False
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[ConfigurationEntity]:
        """Find all configurations."""
        configs = list(self._data.values())
        configs.sort(key=lambda x: x.config_id)
        return self._apply_pagination(configs, limit, offset)
    
    def find_by(self, criteria: Dict[str, Any]) -> List[ConfigurationEntity]:
        """Find configurations by criteria."""
        results = []
        
        for entity in self._data.values():
            if self._match_config_criteria(entity, criteria):
                results.append(entity)
        
        return results
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count configurations."""
        if criteria:
            return len(self.find_by(criteria))
        return len(self._data)
    
    def exists(self, entity_id: Union[str, int]) -> bool:
        """Check if configuration exists."""
        return entity_id in self._data
    
    def get_value(
        self,
        namespace: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get configuration value."""
        config_id = f"{namespace}.{key}"
        
        # Check cache first
        if config_id in self._config_cache:
            return self._config_cache[config_id]
        
        # Load from storage
        entity = self.read(config_id)
        if entity:
            self._config_cache[config_id] = entity.value
            return entity.value
        
        return default
    
    def set_value(
        self,
        namespace: str,
        key: str,
        value: Any,
        config_type: str = 'string',
        description: Optional[str] = None
    ) -> ConfigurationEntity:
        """Set configuration value."""
        config_id = f"{namespace}.{key}"
        
        # Check if exists
        entity = self.read(config_id)
        
        if entity:
            # Update existing
            entity.value = value
            entity.config_type = config_type
            if description:
                entity.description = description
            return self.update(entity)
        else:
            # Create new
            entity = ConfigurationEntity(
                config_id=config_id,
                namespace=namespace,
                key=key,
                value=value,
                config_type=config_type,
                description=description
            )
            return self.create(entity)
    
    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        """Get all configurations in a namespace."""
        configs = {}
        
        for config_id in self._namespace_index.get(namespace, []):
            entity = self.read(config_id)
            if entity:
                configs[entity.key] = entity.value
        
        return configs
    
    def set_namespace(
        self,
        namespace: str,
        configs: Dict[str, Any]
    ) -> List[ConfigurationEntity]:
        """Set multiple configurations in a namespace."""
        entities = []
        
        for key, value in configs.items():
            entity = self.set_value(namespace, key, value)
            entities.append(entity)
        
        return entities
    
    def delete_namespace(self, namespace: str) -> int:
        """Delete all configurations in a namespace."""
        config_ids = self._namespace_index.get(namespace, []).copy()
        
        deleted_count = 0
        for config_id in config_ids:
            if self.delete(config_id):
                deleted_count += 1
        
        return deleted_count
    
    def export_config(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Export configurations."""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'configurations': {}
        }
        
        if namespace:
            # Export specific namespace
            export_data['configurations'][namespace] = self.get_namespace(namespace)
        else:
            # Export all namespaces
            for ns in self._namespace_index.keys():
                export_data['configurations'][ns] = self.get_namespace(ns)
        
        return export_data
    
    def import_config(
        self,
        config_data: Dict[str, Any],
        overwrite: bool = False
    ) -> int:
        """Import configurations."""
        imported_count = 0
        
        configurations = config_data.get('configurations', {})
        
        for namespace, configs in configurations.items():
            for key, value in configs.items():
                config_id = f"{namespace}.{key}"
                
                if self.exists(config_id) and not overwrite:
                    logger.warning(f"Skipping existing config: {config_id}")
                    continue
                
                self.set_value(namespace, key, value)
                imported_count += 1
        
        return imported_count
    
    def get_config_history(self, config_id: str) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        # In production, would maintain actual history
        entity = self.read(config_id)
        
        if not entity:
            return []
        
        return [{
            'timestamp': entity.updated_at,
            'value': entity.value,
            'action': 'current'
        }]
    
    def validate_config(
        self,
        namespace: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate namespace configuration against schema."""
        configs = self.get_namespace(namespace)
        
        errors = []
        warnings = []
        
        # Check required fields
        required = schema.get('required', [])
        for field in required:
            if field not in configs:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        for key, value in configs.items():
            if key in properties:
                expected_type = properties[key].get('type')
                if not self._check_type(value, expected_type):
                    errors.append(f"Invalid type for {key}: expected {expected_type}")
        
        # Check for unknown fields
        for key in configs.keys():
            if key not in properties:
                warnings.append(f"Unknown configuration: {key}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _load_default_configs(self):
        """Load default configurations."""
        defaults = {
            'system.debug': False,
            'system.log_level': 'INFO',
            'model.default_threshold': 0.5,
            'model.window_size': 30,
            'api.rate_limit': 100,
            'api.timeout': 30,
            'monitoring.enabled': True,
            'monitoring.interval': 60
        }
        
        for config_id, value in defaults.items():
            if not self.exists(config_id):
                namespace, key = config_id.rsplit('.', 1)
                self.set_value(namespace, key, value)
    
    def _validate_config_type(self, value: Any, config_type: str):
        """Validate configuration value type."""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        if config_type in type_map:
            expected_type = type_map[config_type]
            if not isinstance(value, expected_type):
                raise RepositoryException(
                    f"Invalid type: expected {config_type}, got {type(value).__name__}"
                )
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_checks = {
            'string': lambda v: isinstance(v, str),
            'integer': lambda v: isinstance(v, int),
            'number': lambda v: isinstance(v, (int, float)),
            'boolean': lambda v: isinstance(v, bool),
            'array': lambda v: isinstance(v, list),
            'object': lambda v: isinstance(v, dict)
        }
        
        if expected_type in type_checks:
            return type_checks[expected_type](value)
        
        return True
    
    def _match_config_criteria(
        self,
        entity: ConfigurationEntity,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if configuration matches criteria."""
        for key, value in criteria.items():
            if key == 'namespace' and entity.namespace != value:
                return False
            elif key == 'config_type' and entity.config_type != value:
                return False
            elif key == 'key' and entity.key != value:
                return False
        
        return True
    
    def _create_connection(self):
        """Create connection."""
        return True
    
    def _close_connection(self):
        """Close connection."""
        self._save_data()

from collections import defaultdict