"""Model versioning and metadata management for IoT Anomaly Detector."""

import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .logging_config import get_logger
from .security_utils import validate_file_path, secure_json_load, sanitize_error_message


class ModelMetadata:
    """Manages model versioning and metadata tracking."""
    
    def __init__(self, model_directory: str = "saved_models"):
        """Initialize model metadata manager.
        
        Parameters
        ----------
        model_directory : str
            Directory where models and metadata are stored
        """
        self.model_directory = Path(model_directory)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)
        
    def generate_model_hash(self, model_path: str) -> str:
        """Generate hash for model file to track changes.
        
        Parameters
        ----------
        model_path : str
            Path to the model file
            
        Returns
        -------
        str
            SHA256 hash of the model file
        """
        try:
            # Validate the model file path for security
            validated_path = validate_file_path(model_path)
            
            sha256_hash = hashlib.sha256()
            with open(validated_path, "rb") as f:
                # Read file in chunks to handle large models
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            self.logger.error(f"Failed to generate model hash: {sanitized_error}")
            raise
    
    def create_metadata(
        self,
        model_path: str,
        training_params: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for a trained model.
        
        Parameters
        ----------
        model_path : str
            Path to the trained model file
        training_params : Dict[str, Any]
            Training parameters used (epochs, batch_size, etc.)
        performance_metrics : Dict[str, float], optional
            Performance metrics from evaluation
        dataset_info : Dict[str, Any], optional
            Information about the training dataset
        version : str, optional
            Model version. If None, auto-generated based on timestamp
            
        Returns
        -------
        Dict[str, Any]
            Complete metadata dictionary
        """
        model_file = Path(model_path)
        
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v{timestamp}"
        
        metadata = {
            "version": version,
            "created_at": datetime.datetime.now().isoformat(),
            "model_info": {
                "file_path": str(model_file.absolute()),
                "file_name": model_file.name,
                "file_size_bytes": model_file.stat().st_size if model_file.exists() else 0,
                "model_hash": self.generate_model_hash(model_path) if model_file.exists() else None
            },
            "training_params": training_params,
            "performance_metrics": performance_metrics or {},
            "dataset_info": dataset_info or {},
            "framework_info": {
                "tensorflow_version": self._get_tensorflow_version(),
                "python_version": self._get_python_version()
            }
        }
        
        self.logger.info(f"Created metadata for model {version}", extra={
            "model_version": version,
            "model_hash": metadata["model_info"]["model_hash"][:8] if metadata["model_info"]["model_hash"] else None,
            "training_epochs": training_params.get("epochs", "unknown")
        })
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], metadata_path: Optional[str] = None) -> str:
        """Save model metadata to JSON file.
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata dictionary to save
        metadata_path : str, optional
            Path to save metadata. If None, auto-generated based on version
            
        Returns
        -------
        str
            Path where metadata was saved
        """
        if metadata_path is None:
            version = metadata["version"]
            metadata_path = self.model_directory / f"metadata_{version}.json"
        
        metadata_file = Path(metadata_path)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {metadata_file}")
        return str(metadata_file)
    
    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load model metadata from JSON file.
        
        Parameters
        ----------
        metadata_path : str
            Path to metadata file
            
        Returns
        -------
        Dict[str, Any]
            Loaded metadata dictionary
        """
        try:
            # Use secure JSON loading with path validation and size limits
            metadata = secure_json_load(metadata_path, max_size_mb=10.0)
            
            self.logger.info(f"Loaded metadata from {sanitize_error_message(metadata_path)}")
            return metadata
        except Exception as e:
            sanitized_error = sanitize_error_message(str(e))
            self.logger.error(f"Failed to load metadata: {sanitized_error}")
            raise
    
    def list_model_versions(self) -> list[Dict[str, Any]]:
        """List all available model versions with their metadata.
        
        Returns
        -------
        list[Dict[str, Any]]
            List of metadata summaries for all available models
        """
        metadata_files = list(self.model_directory.glob("metadata_*.json"))
        versions = []
        
        for metadata_file in metadata_files:
            try:
                metadata = self.load_metadata(metadata_file)
                summary = {
                    "version": metadata["version"],
                    "created_at": metadata["created_at"],
                    "model_file": metadata["model_info"]["file_name"],
                    "training_epochs": metadata["training_params"].get("epochs", "unknown"),
                    "performance": metadata.get("performance_metrics", {})
                }
                versions.append(summary)
            except Exception as e:
                self.logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions
    
    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions.
        
        Parameters
        ----------
        version1, version2 : str
            Model versions to compare
            
        Returns
        -------
        Dict[str, Any]
            Comparison results showing differences in parameters and performance
        """
        metadata1_path = self.model_directory / f"metadata_{version1}.json"
        metadata2_path = self.model_directory / f"metadata_{version2}.json"
        
        metadata1 = self.load_metadata(metadata1_path)
        metadata2 = self.load_metadata(metadata2_path)
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "parameter_differences": self._compare_dicts(
                metadata1["training_params"], 
                metadata2["training_params"]
            ),
            "performance_differences": self._compare_dicts(
                metadata1.get("performance_metrics", {}),
                metadata2.get("performance_metrics", {})
            ),
            "model_hash_different": (
                metadata1["model_info"]["model_hash"] != 
                metadata2["model_info"]["model_hash"]
            )
        }
        
        return comparison
    
    def _compare_dicts(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two dictionaries and return differences."""
        differences = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    "version1": val1,
                    "version2": val2
                }
        
        return differences
    
    def _get_tensorflow_version(self) -> str:
        """Get TensorFlow version."""
        try:
            import tensorflow as tf
            return tf.__version__
        except ImportError:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def save_model_with_metadata(
    model_path: str,
    training_params: Dict[str, Any],
    performance_metrics: Optional[Dict[str, float]] = None,
    dataset_info: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None,
    metadata_directory: str = "saved_models"
) -> tuple[str, str]:
    """Convenience function to save model metadata.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model file
    training_params : Dict[str, Any]
        Training parameters used
    performance_metrics : Dict[str, float], optional
        Performance metrics from evaluation
    dataset_info : Dict[str, Any], optional
        Information about the training dataset
    version : str, optional
        Model version
    metadata_directory : str
        Directory for metadata storage
        
    Returns
    -------
    tuple[str, str]
        Paths to model and metadata files
    """
    metadata_manager = ModelMetadata(metadata_directory)
    
    metadata = metadata_manager.create_metadata(
        model_path=model_path,
        training_params=training_params,
        performance_metrics=performance_metrics,
        dataset_info=dataset_info,
        version=version
    )
    
    metadata_path = metadata_manager.save_metadata(metadata)
    
    return model_path, metadata_path