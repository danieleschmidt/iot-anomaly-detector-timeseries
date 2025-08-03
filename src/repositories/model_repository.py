"""
Model Repository

Repository for managing machine learning model artifacts and metadata.
"""

import logging
import json
import shutil
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import hashlib

from .base_repository import FileBasedRepository, RepositoryException

logger = logging.getLogger(__name__)


class ModelEntity:
    """Entity representing a machine learning model."""
    
    def __init__(
        self,
        model_id: str,
        version: str,
        name: str,
        model_path: str,
        metadata: Dict[str, Any],
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None
    ):
        self.model_id = model_id
        self.version = version
        self.name = name
        self.model_path = model_path
        self.metadata = metadata
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'name': self.name,
            'model_path': self.model_path,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelEntity':
        """Create entity from dictionary."""
        return cls(**data)


class ModelRepository(FileBasedRepository[ModelEntity]):
    """
    Repository for model persistence and retrieval.
    
    Manages model artifacts, versions, and metadata in a structured way.
    """
    
    def __init__(
        self,
        base_path: str = "saved_models",
        storage_backend: str = "filesystem"
    ):
        """
        Initialize model repository.
        
        Args:
            base_path: Base directory for model storage
            storage_backend: Storage backend type
        """
        super().__init__(base_path=base_path, file_format='json')
        self.storage_backend = storage_backend
        self.models_dir = self.base_path / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.base_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def create(self, entity: ModelEntity) -> ModelEntity:
        """
        Create a new model entry.
        
        Args:
            entity: Model entity to create
            
        Returns:
            Created model entity
        """
        # Generate model ID if not provided
        if not entity.model_id:
            entity.model_id = self._generate_model_id(entity.name, entity.version)
        
        # Check if model already exists
        if self.exists(entity.model_id):
            raise RepositoryException(f"Model {entity.model_id} already exists")
        
        # Copy model file to repository
        src_path = Path(entity.model_path)
        if not src_path.exists():
            raise RepositoryException(f"Model file not found: {entity.model_path}")
        
        dest_path = self.models_dir / f"{entity.model_id}.model"
        shutil.copy2(src_path, dest_path)
        entity.model_path = str(dest_path)
        
        # Add audit fields
        entity.created_at = datetime.now().isoformat()
        entity.updated_at = entity.created_at
        
        # Save metadata
        self._save_model_metadata(entity)
        
        # Add to in-memory data
        self._data[entity.model_id] = entity
        
        logger.info(f"Created model: {entity.model_id}")
        return entity
    
    def read(self, entity_id: Union[str, int]) -> Optional[ModelEntity]:
        """
        Read a model by ID.
        
        Args:
            entity_id: Model identifier
            
        Returns:
            Model entity if found
        """
        # Check in-memory data first
        if entity_id in self._data:
            return self._data[entity_id]
        
        # Try to load from metadata
        metadata_path = self.metadata_dir / f"{entity_id}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            entity = ModelEntity.from_dict(data)
            self._data[entity_id] = entity
            return entity
        
        return None
    
    def update(self, entity: ModelEntity) -> ModelEntity:
        """
        Update a model entry.
        
        Args:
            entity: Model entity with updated values
            
        Returns:
            Updated model entity
        """
        if not self.exists(entity.model_id):
            raise RepositoryException(f"Model {entity.model_id} not found")
        
        # Update timestamp
        entity.updated_at = datetime.now().isoformat()
        
        # Update model file if path changed
        existing = self._data.get(entity.model_id)
        if existing and existing.model_path != entity.model_path:
            src_path = Path(entity.model_path)
            if src_path.exists():
                dest_path = self.models_dir / f"{entity.model_id}.model"
                shutil.copy2(src_path, dest_path)
                entity.model_path = str(dest_path)
        
        # Save updated metadata
        self._save_model_metadata(entity)
        
        # Update in-memory data
        self._data[entity.model_id] = entity
        
        logger.info(f"Updated model: {entity.model_id}")
        return entity
    
    def delete(self, entity_id: Union[str, int]) -> bool:
        """
        Delete a model by ID.
        
        Args:
            entity_id: Model identifier
            
        Returns:
            True if deleted
        """
        if not self.exists(entity_id):
            return False
        
        # Delete model file
        model_path = self.models_dir / f"{entity_id}.model"
        if model_path.exists():
            model_path.unlink()
        
        # Delete metadata
        metadata_path = self.metadata_dir / f"{entity_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from in-memory data
        if entity_id in self._data:
            del self._data[entity_id]
        
        logger.info(f"Deleted model: {entity_id}")
        return True
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[ModelEntity]:
        """
        Find all models with pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of model entities
        """
        # Load all metadata files
        self._load_all_metadata()
        
        # Get all models
        models = list(self._data.values())
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return self._apply_pagination(models, limit, offset)
    
    def find_by(self, criteria: Dict[str, Any]) -> List[ModelEntity]:
        """
        Find models matching criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching models
        """
        # Load all metadata files
        self._load_all_metadata()
        
        # Filter models
        results = []
        for entity in self._data.values():
            if self._match_model_criteria(entity, criteria):
                results.append(entity)
        
        return results
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count models matching criteria.
        
        Args:
            criteria: Optional filter criteria
            
        Returns:
            Number of matching models
        """
        if criteria:
            return len(self.find_by(criteria))
        else:
            self._load_all_metadata()
            return len(self._data)
    
    def exists(self, entity_id: Union[str, int]) -> bool:
        """
        Check if model exists.
        
        Args:
            entity_id: Model identifier
            
        Returns:
            True if exists
        """
        # Check in-memory data
        if entity_id in self._data:
            return True
        
        # Check metadata file
        metadata_path = self.metadata_dir / f"{entity_id}.json"
        return metadata_path.exists()
    
    def find_by_version(self, version: str) -> Optional[ModelEntity]:
        """
        Find model by version.
        
        Args:
            version: Model version
            
        Returns:
            Model entity if found
        """
        results = self.find_by({'version': version})
        return results[0] if results else None
    
    def find_latest_version(self, name: Optional[str] = None) -> Optional[ModelEntity]:
        """
        Find latest model version.
        
        Args:
            name: Optional model name filter
            
        Returns:
            Latest model entity
        """
        criteria = {'name': name} if name else {}
        models = self.find_by(criteria)
        
        if not models:
            return None
        
        # Sort by creation date and return latest
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models[0]
    
    def find_by_deployment_target(self, target: str) -> List[ModelEntity]:
        """
        Find models deployed to specific target.
        
        Args:
            target: Deployment target (production, staging, etc.)
            
        Returns:
            List of deployed models
        """
        return self.find_by({'metadata.deployment_target': target})
    
    def get_model_history(self, name: str) -> List[ModelEntity]:
        """
        Get version history for a model.
        
        Args:
            name: Model name
            
        Returns:
            List of model versions
        """
        models = self.find_by({'name': name})
        models.sort(key=lambda x: x.created_at)
        return models
    
    def archive_model(self, model_id: str) -> bool:
        """
        Archive a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if archived
        """
        entity = self.read(model_id)
        if not entity:
            return False
        
        # Move to archive directory
        archive_dir = self.base_path / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model file
        src_model = Path(entity.model_path)
        if src_model.exists():
            dest_model = archive_dir / src_model.name
            shutil.move(str(src_model), str(dest_model))
        
        # Update metadata
        entity.metadata['archived'] = True
        entity.metadata['archived_at'] = datetime.now().isoformat()
        self.update(entity)
        
        logger.info(f"Archived model: {model_id}")
        return True
    
    def restore_model(self, model_id: str) -> bool:
        """
        Restore an archived model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if restored
        """
        entity = self.read(model_id)
        if not entity or not entity.metadata.get('archived'):
            return False
        
        # Move from archive directory
        archive_dir = self.base_path / "archive"
        model_file = archive_dir / f"{model_id}.model"
        
        if model_file.exists():
            dest_model = self.models_dir / model_file.name
            shutil.move(str(model_file), str(dest_model))
            entity.model_path = str(dest_model)
        
        # Update metadata
        entity.metadata['archived'] = False
        entity.metadata['restored_at'] = datetime.now().isoformat()
        self.update(entity)
        
        logger.info(f"Restored model: {model_id}")
        return True
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Storage statistics
        """
        total_size = 0
        model_count = 0
        
        # Calculate total size
        for model_file in self.models_dir.glob("*.model"):
            total_size += model_file.stat().st_size
            model_count += 1
        
        # Get archive stats
        archive_dir = self.base_path / "archive"
        archive_size = 0
        archive_count = 0
        
        if archive_dir.exists():
            for model_file in archive_dir.glob("*.model"):
                archive_size += model_file.stat().st_size
                archive_count += 1
        
        return {
            'total_models': model_count,
            'total_size_mb': total_size / (1024 * 1024),
            'archived_models': archive_count,
            'archive_size_mb': archive_size / (1024 * 1024),
            'storage_path': str(self.base_path)
        }
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID."""
        hash_input = f"{name}_{version}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _save_model_metadata(self, entity: ModelEntity) -> None:
        """Save model metadata to file."""
        metadata_path = self.metadata_dir / f"{entity.model_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(entity.to_dict(), f, indent=2)
    
    def _load_all_metadata(self) -> None:
        """Load all metadata files into memory."""
        for metadata_file in self.metadata_dir.glob("*.json"):
            model_id = metadata_file.stem
            if model_id not in self._data:
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    self._data[model_id] = ModelEntity.from_dict(data)
                except Exception as e:
                    logger.error(f"Failed to load metadata {metadata_file}: {e}")
    
    def _match_model_criteria(
        self,
        entity: ModelEntity,
        criteria: Dict[str, Any]
    ) -> bool:
        """Check if model matches criteria."""
        entity_dict = entity.to_dict()
        
        for key, value in criteria.items():
            # Handle nested keys (e.g., metadata.deployment_target)
            if '.' in key:
                parts = key.split('.')
                current = entity_dict
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False
                if current != value:
                    return False
            else:
                # Simple key match
                if key not in entity_dict or entity_dict[key] != value:
                    return False
        
        return True