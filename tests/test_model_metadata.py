import json
import pytest
import tempfile
from pathlib import Path
import hashlib

pytest.importorskip("tensorflow")

from src.model_metadata import ModelMetadata, save_model_with_metadata


def create_dummy_model_file(path: Path) -> str:
    """Create a dummy model file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"dummy_model_content_for_testing"
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_model_metadata_creation(tmp_path):
    """Test basic model metadata creation."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    # Create dummy model file
    model_path = tmp_path / "test_model.h5"
    expected_hash = create_dummy_model_file(model_path)
    
    training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    performance_metrics = {
        "final_loss": 0.05,
        "accuracy": 0.95
    }
    
    metadata = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params=training_params,
        performance_metrics=performance_metrics,
        version="test_v1"
    )
    
    assert metadata["version"] == "test_v1"
    assert metadata["training_params"] == training_params
    assert metadata["performance_metrics"] == performance_metrics
    assert metadata["model_info"]["model_hash"] == expected_hash
    assert metadata["model_info"]["file_size_bytes"] > 0


def test_metadata_save_and_load(tmp_path):
    """Test saving and loading metadata."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    model_path = tmp_path / "test_model.h5"
    create_dummy_model_file(model_path)
    
    training_params = {"epochs": 5, "batch_size": 16}
    metadata = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params=training_params,
        version="test_v2"
    )
    
    # Save metadata
    metadata_path = metadata_manager.save_metadata(metadata)
    assert Path(metadata_path).exists()
    
    # Load metadata
    loaded_metadata = metadata_manager.load_metadata(metadata_path)
    assert loaded_metadata["version"] == "test_v2"
    assert loaded_metadata["training_params"] == training_params


def test_list_model_versions(tmp_path):
    """Test listing model versions."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    # Create multiple model versions
    for i in range(3):
        model_path = tmp_path / f"model_{i}.h5"
        create_dummy_model_file(model_path)
        
        metadata = metadata_manager.create_metadata(
            model_path=str(model_path),
            training_params={"epochs": i + 1},
            version=f"v{i}"
        )
        metadata_manager.save_metadata(metadata)
    
    versions = metadata_manager.list_model_versions()
    assert len(versions) == 3
    
    # Check that versions are sorted by creation date (newest first)
    assert versions[0]["version"] == "v2"
    assert versions[1]["version"] == "v1"
    assert versions[2]["version"] == "v0"


def test_compare_models(tmp_path):
    """Test model comparison functionality."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    # Create two different models
    model1_path = tmp_path / "model1.h5"
    model2_path = tmp_path / "model2.h5"
    create_dummy_model_file(model1_path)
    create_dummy_model_file(model2_path)
    
    # Create metadata with different parameters
    metadata1 = metadata_manager.create_metadata(
        model_path=str(model1_path),
        training_params={"epochs": 10, "batch_size": 32},
        performance_metrics={"loss": 0.1},
        version="v1"
    )
    
    metadata2 = metadata_manager.create_metadata(
        model_path=str(model2_path),
        training_params={"epochs": 20, "batch_size": 32},
        performance_metrics={"loss": 0.05},
        version="v2"
    )
    
    metadata_manager.save_metadata(metadata1)
    metadata_manager.save_metadata(metadata2)
    
    comparison = metadata_manager.compare_models("v1", "v2")
    
    assert comparison["version1"] == "v1"
    assert comparison["version2"] == "v2"
    assert "epochs" in comparison["parameter_differences"]
    assert comparison["parameter_differences"]["epochs"]["version1"] == 10
    assert comparison["parameter_differences"]["epochs"]["version2"] == 20
    assert "loss" in comparison["performance_differences"]


def test_save_model_with_metadata_convenience_function(tmp_path):
    """Test the convenience function for saving model with metadata."""
    model_path = tmp_path / "convenience_model.h5"
    create_dummy_model_file(model_path)
    
    training_params = {"epochs": 15, "learning_rate": 0.001}
    performance_metrics = {"accuracy": 0.92}
    dataset_info = {"num_samples": 1000, "features": 3}
    
    model_path_result, metadata_path = save_model_with_metadata(
        model_path=str(model_path),
        training_params=training_params,
        performance_metrics=performance_metrics,
        dataset_info=dataset_info,
        version="convenience_test",
        metadata_directory=str(tmp_path)
    )
    
    assert Path(metadata_path).exists()
    
    # Load and verify metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    assert metadata["version"] == "convenience_test"
    assert metadata["training_params"] == training_params
    assert metadata["performance_metrics"] == performance_metrics
    assert metadata["dataset_info"] == dataset_info


def test_model_hash_generation(tmp_path):
    """Test model hash generation for file integrity."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    # Create two identical files
    model1_path = tmp_path / "model1.h5"
    model2_path = tmp_path / "model2.h5"
    
    content = b"identical_content"
    model1_path.write_bytes(content)
    model2_path.write_bytes(content)
    
    hash1 = metadata_manager.generate_model_hash(str(model1_path))
    hash2 = metadata_manager.generate_model_hash(str(model2_path))
    
    assert hash1 == hash2
    
    # Modify one file
    model2_path.write_bytes(b"different_content")
    hash2_modified = metadata_manager.generate_model_hash(str(model2_path))
    
    assert hash1 != hash2_modified


def test_missing_model_file_error(tmp_path):
    """Test error handling for missing model files."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    non_existent_path = tmp_path / "does_not_exist.h5"
    
    with pytest.raises(FileNotFoundError):
        metadata_manager.generate_model_hash(str(non_existent_path))


def test_automatic_version_generation(tmp_path):
    """Test automatic version generation when not specified."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    model_path = tmp_path / "auto_version_model.h5"
    create_dummy_model_file(model_path)
    
    # Don't specify version - should auto-generate
    metadata = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params={"epochs": 1}
    )
    
    # Check that version was auto-generated with timestamp format
    assert metadata["version"].startswith("v")
    assert len(metadata["version"]) > 10  # Should include timestamp


def test_metadata_framework_info(tmp_path):
    """Test that framework information is captured."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    model_path = tmp_path / "framework_test_model.h5"
    create_dummy_model_file(model_path)
    
    metadata = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params={"epochs": 1}
    )
    
    framework_info = metadata["framework_info"]
    assert "tensorflow_version" in framework_info
    assert "python_version" in framework_info
    assert framework_info["python_version"] != "unknown"


def test_edge_case_empty_performance_metrics(tmp_path):
    """Test handling of empty or missing performance metrics."""
    metadata_manager = ModelMetadata(str(tmp_path))
    
    model_path = tmp_path / "empty_metrics_model.h5"
    create_dummy_model_file(model_path)
    
    # Test with None performance metrics
    metadata = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params={"epochs": 1},
        performance_metrics=None
    )
    
    assert metadata["performance_metrics"] == {}
    
    # Test with empty dict
    metadata2 = metadata_manager.create_metadata(
        model_path=str(model_path),
        training_params={"epochs": 1},
        performance_metrics={}
    )
    
    assert metadata2["performance_metrics"] == {}