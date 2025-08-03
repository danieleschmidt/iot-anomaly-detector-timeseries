"""
Base Repository

Abstract base class for all repository implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from datetime import datetime
from pathlib import Path
import json
import pickle
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RepositoryException(Exception):
    """Custom exception for repository operations."""
    pass


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository providing common data access patterns.
    
    This class defines the interface that all repositories must implement
    and provides common functionality for data persistence.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the repository.
        
        Args:
            connection_string: Database connection string
            config: Repository configuration
        """
        self.connection_string = connection_string
        self.config = config or {}
        self._connection = None
        self._transaction_active = False
        
    @abstractmethod
    def create(self, entity: T) -> T:
        """
        Create a new entity.
        
        Args:
            entity: Entity to create
            
        Returns:
            Created entity with generated ID
        """
        pass
    
    @abstractmethod
    def read(self, entity_id: Union[str, int]) -> Optional[T]:
        """
        Read an entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update(self, entity: T) -> T:
        """
        Update an existing entity.
        
        Args:
            entity: Entity with updated values
            
        Returns:
            Updated entity
        """
        pass
    
    @abstractmethod
    def delete(self, entity_id: Union[str, int]) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if deleted, False otherwise
        """
        pass
    
    @abstractmethod
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """
        Find all entities with optional pagination.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of entities
        """
        pass
    
    @abstractmethod
    def find_by(self, criteria: Dict[str, Any]) -> List[T]:
        """
        Find entities matching criteria.
        
        Args:
            criteria: Search criteria
            
        Returns:
            List of matching entities
        """
        pass
    
    @abstractmethod
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count entities matching criteria.
        
        Args:
            criteria: Optional filter criteria
            
        Returns:
            Number of matching entities
        """
        pass
    
    @abstractmethod
    def exists(self, entity_id: Union[str, int]) -> bool:
        """
        Check if entity exists.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if exists, False otherwise
        """
        pass
    
    def batch_create(self, entities: List[T]) -> List[T]:
        """
        Create multiple entities in batch.
        
        Args:
            entities: List of entities to create
            
        Returns:
            List of created entities
        """
        created = []
        for entity in entities:
            try:
                created.append(self.create(entity))
            except Exception as e:
                logger.error(f"Failed to create entity: {e}")
                if self.config.get('fail_fast', True):
                    raise
        return created
    
    def batch_update(self, entities: List[T]) -> List[T]:
        """
        Update multiple entities in batch.
        
        Args:
            entities: List of entities to update
            
        Returns:
            List of updated entities
        """
        updated = []
        for entity in entities:
            try:
                updated.append(self.update(entity))
            except Exception as e:
                logger.error(f"Failed to update entity: {e}")
                if self.config.get('fail_fast', True):
                    raise
        return updated
    
    def batch_delete(self, entity_ids: List[Union[str, int]]) -> int:
        """
        Delete multiple entities in batch.
        
        Args:
            entity_ids: List of entity identifiers
            
        Returns:
            Number of deleted entities
        """
        deleted_count = 0
        for entity_id in entity_ids:
            try:
                if self.delete(entity_id):
                    deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete entity {entity_id}: {e}")
                if self.config.get('fail_fast', True):
                    raise
        return deleted_count
    
    @contextmanager
    def transaction(self):
        """
        Context manager for transactions.
        
        Usage:
            with repo.transaction():
                repo.create(entity1)
                repo.update(entity2)
        """
        self.begin_transaction()
        try:
            yield self
            self.commit_transaction()
        except Exception as e:
            self.rollback_transaction()
            raise e
    
    def begin_transaction(self):
        """Begin a new transaction."""
        if self._transaction_active:
            raise RepositoryException("Transaction already active")
        self._transaction_active = True
        logger.debug("Transaction started")
    
    def commit_transaction(self):
        """Commit the current transaction."""
        if not self._transaction_active:
            raise RepositoryException("No active transaction")
        self._transaction_active = False
        logger.debug("Transaction committed")
    
    def rollback_transaction(self):
        """Rollback the current transaction."""
        if not self._transaction_active:
            raise RepositoryException("No active transaction")
        self._transaction_active = False
        logger.debug("Transaction rolled back")
    
    def connect(self):
        """Establish connection to data store."""
        if self._connection:
            return
        
        # Implementation depends on specific data store
        logger.info("Establishing connection to data store")
        self._connection = self._create_connection()
    
    def disconnect(self):
        """Close connection to data store."""
        if self._connection:
            self._close_connection()
            self._connection = None
            logger.info("Disconnected from data store")
    
    def is_connected(self) -> bool:
        """Check if connected to data store."""
        return self._connection is not None
    
    @abstractmethod
    def _create_connection(self):
        """Create connection to data store."""
        pass
    
    @abstractmethod
    def _close_connection(self):
        """Close connection to data store."""
        pass
    
    def _serialize(self, entity: T, format: str = 'json') -> bytes:
        """
        Serialize entity for storage.
        
        Args:
            entity: Entity to serialize
            format: Serialization format
            
        Returns:
            Serialized entity
        """
        if format == 'json':
            return json.dumps(entity, default=str).encode()
        elif format == 'pickle':
            return pickle.dumps(entity)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _deserialize(self, data: bytes, format: str = 'json') -> T:
        """
        Deserialize entity from storage.
        
        Args:
            data: Serialized data
            format: Serialization format
            
        Returns:
            Deserialized entity
        """
        if format == 'json':
            return json.loads(data.decode())
        elif format == 'pickle':
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _validate_entity(self, entity: T) -> bool:
        """
        Validate entity before persistence.
        
        Args:
            entity: Entity to validate
            
        Returns:
            True if valid
            
        Raises:
            RepositoryException: If entity is invalid
        """
        if entity is None:
            raise RepositoryException("Entity cannot be None")
        return True
    
    def _add_audit_fields(self, entity: Dict[str, Any], operation: str) -> Dict[str, Any]:
        """
        Add audit fields to entity.
        
        Args:
            entity: Entity dictionary
            operation: Operation type (create, update, delete)
            
        Returns:
            Entity with audit fields
        """
        now = datetime.now().isoformat()
        
        if operation == 'create':
            entity['created_at'] = now
            entity['updated_at'] = now
        elif operation == 'update':
            entity['updated_at'] = now
        elif operation == 'delete':
            entity['deleted_at'] = now
        
        return entity
    
    def _apply_pagination(
        self,
        results: List[T],
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """
        Apply pagination to results.
        
        Args:
            results: Full result set
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Paginated results
        """
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]
        return results
    
    def _match_criteria(self, entity: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """
        Check if entity matches search criteria.
        
        Args:
            entity: Entity to check
            criteria: Search criteria
            
        Returns:
            True if matches all criteria
        """
        for key, value in criteria.items():
            if key not in entity:
                return False
            
            # Handle different comparison operators
            if isinstance(value, dict):
                if '$eq' in value and entity[key] != value['$eq']:
                    return False
                if '$ne' in value and entity[key] == value['$ne']:
                    return False
                if '$gt' in value and entity[key] <= value['$gt']:
                    return False
                if '$gte' in value and entity[key] < value['$gte']:
                    return False
                if '$lt' in value and entity[key] >= value['$lt']:
                    return False
                if '$lte' in value and entity[key] > value['$lte']:
                    return False
                if '$in' in value and entity[key] not in value['$in']:
                    return False
                if '$nin' in value and entity[key] in value['$nin']:
                    return False
            else:
                # Simple equality check
                if entity[key] != value:
                    return False
        
        return True


class FileBasedRepository(BaseRepository[T]):
    """
    File-based repository implementation for simple persistence.
    
    This is a base class for repositories that store data in files.
    """
    
    def __init__(
        self,
        base_path: str,
        file_format: str = 'json',
        **kwargs
    ):
        """
        Initialize file-based repository.
        
        Args:
            base_path: Base directory for data files
            file_format: File format (json, pickle)
        """
        super().__init__(**kwargs)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.file_format = file_format
        self._data: Dict[str, T] = {}
        self._load_data()
    
    def _create_connection(self):
        """Create connection (load data from files)."""
        return True
    
    def _close_connection(self):
        """Close connection (save data to files)."""
        self._save_data()
    
    def _load_data(self):
        """Load all data from files."""
        extension = '.json' if self.file_format == 'json' else '.pkl'
        
        for file_path in self.base_path.glob(f'*{extension}'):
            entity_id = file_path.stem
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                entity = self._deserialize(data, self.file_format)
                self._data[entity_id] = entity
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    def _save_data(self):
        """Save all data to files."""
        extension = '.json' if self.file_format == 'json' else '.pkl'
        
        for entity_id, entity in self._data.items():
            file_path = self.base_path / f"{entity_id}{extension}"
            try:
                data = self._serialize(entity, self.file_format)
                with open(file_path, 'wb') as f:
                    f.write(data)
            except Exception as e:
                logger.error(f"Failed to save {file_path}: {e}")
    
    def _save_entity(self, entity_id: str, entity: T):
        """Save single entity to file."""
        extension = '.json' if self.file_format == 'json' else '.pkl'
        file_path = self.base_path / f"{entity_id}{extension}"
        
        data = self._serialize(entity, self.file_format)
        with open(file_path, 'wb') as f:
            f.write(data)
    
    def _delete_entity_file(self, entity_id: str):
        """Delete entity file."""
        extension = '.json' if self.file_format == 'json' else '.pkl'
        file_path = self.base_path / f"{entity_id}{extension}"
        
        if file_path.exists():
            file_path.unlink()