"""Comprehensive tests for model_manager CLI utilities."""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import sys
import argparse

from src.model_manager import (
    list_models, show_metadata, compare_models, cleanup_old_models, main
)
from src.model_metadata import ModelMetadata


class TestModelManagerCLI:
    """Test suite for model manager CLI functionality."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "version": "v20240120_143022",
            "created_at": "2024-01-20T14:30:22.123456",
            "model_info": {
                "file_path": "/path/to/model.h5",
                "file_name": "model.h5", 
                "file_size_bytes": 1024,
                "model_hash": "abcd1234"
            },
            "training_params": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "performance_metrics": {
                "loss": 0.025,
                "accuracy": 0.95
            }
        }
    
    @pytest.fixture
    def mock_args(self, temp_model_dir):
        """Create mock arguments object."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        return args


class TestListModels:
    """Test list_models functionality."""
    
    def test_list_models_empty_directory(self, temp_model_dir, capsys):
        """Test listing models when no models exist."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        
        list_models(args)
        
        captured = capsys.readouterr()
        assert "No model versions found." in captured.out
    
    def test_list_models_with_models(self, temp_model_dir, sample_metadata, capsys):
        """Test listing models when models exist."""
        # Create sample metadata files
        metadata_file1 = Path(temp_model_dir) / "metadata_v20240120_143022.json"
        metadata_file2 = Path(temp_model_dir) / "metadata_v20240121_091234.json"
        
        metadata2 = sample_metadata.copy()
        metadata2["version"] = "v20240121_091234"
        metadata2["created_at"] = "2024-01-21T09:12:34.567890"
        metadata2["training_params"]["epochs"] = 75
        
        with open(metadata_file1, 'w') as f:
            json.dump(sample_metadata, f)
        with open(metadata_file2, 'w') as f:
            json.dump(metadata2, f)
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        
        list_models(args)
        
        captured = capsys.readouterr()
        assert "Found 2 model versions:" in captured.out
        assert "v20240120_143022" in captured.out
        assert "v20240121_091234" in captured.out
        assert "2024-01-20 14:30:22" in captured.out
        assert "50" in captured.out  # epochs
        assert "75" in captured.out  # epochs
    
    @patch('src.model_manager.ModelMetadata')
    def test_list_models_metadata_error(self, mock_metadata_class, temp_model_dir):
        """Test list_models handles metadata loading errors gracefully."""
        mock_metadata = MagicMock()
        mock_metadata.list_model_versions.side_effect = Exception("Metadata corrupted")
        mock_metadata_class.return_value = mock_metadata
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        
        with pytest.raises(Exception, match="Metadata corrupted"):
            list_models(args)


class TestShowMetadata:
    """Test show_metadata functionality."""
    
    def test_show_metadata_existing_version(self, temp_model_dir, sample_metadata, capsys):
        """Test showing metadata for existing version."""
        metadata_file = Path(temp_model_dir) / "metadata_v20240120_143022.json"
        with open(metadata_file, 'w') as f:
            json.dump(sample_metadata, f)
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version = "v20240120_143022"
        
        show_metadata(args)
        
        captured = capsys.readouterr()
        output_json = json.loads(captured.out)
        assert output_json["version"] == "v20240120_143022"
        assert output_json["training_params"]["epochs"] == 50
    
    def test_show_metadata_nonexistent_version(self, temp_model_dir, capsys):
        """Test showing metadata for non-existent version exits with error."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version = "nonexistent_version"
        
        with pytest.raises(SystemExit) as exc_info:
            show_metadata(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Model version 'nonexistent_version' not found." in captured.out
    
    @patch('src.model_manager.ModelMetadata')
    def test_show_metadata_load_error(self, mock_metadata_class, temp_model_dir):
        """Test show_metadata handles loading errors."""
        mock_metadata = MagicMock()
        mock_metadata.load_metadata.side_effect = FileNotFoundError("File not found")
        mock_metadata_class.return_value = mock_metadata
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version = "test_version"
        
        with pytest.raises(SystemExit):
            show_metadata(args)


class TestCompareModels:
    """Test compare_models functionality."""
    
    def test_compare_models_successful(self, temp_model_dir, capsys):
        """Test successful model comparison."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version1 = "v1"
        args.version2 = "v2"
        
        mock_comparison = {
            'parameter_differences': {
                'epochs': {'version1': 50, 'version2': 75},
                'learning_rate': {'version1': 0.001, 'version2': 0.002}
            },
            'performance_differences': {
                'loss': {'version1': 0.025, 'version2': 0.020},
                'accuracy': {'version1': 0.95, 'version2': 0.97}
            },
            'model_hash_different': True
        }
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.compare_models.return_value = mock_comparison
            mock_metadata_class.return_value = mock_metadata
            
            compare_models(args)
            
            captured = capsys.readouterr()
            assert "Comparing v1 vs v2" in captured.out
            assert "epochs: 50 -> 75" in captured.out
            assert "learning_rate: 0.001 -> 0.002" in captured.out
            assert "loss: 0.025 -> 0.020" in captured.out
            assert "Model Hash Different: True" in captured.out
    
    def test_compare_models_no_differences(self, temp_model_dir, capsys):
        """Test model comparison with no differences."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version1 = "v1"
        args.version2 = "v2"
        
        mock_comparison = {
            'parameter_differences': {},
            'performance_differences': {},
            'model_hash_different': False
        }
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.compare_models.return_value = mock_comparison
            mock_metadata_class.return_value = mock_metadata
            
            compare_models(args)
            
            captured = capsys.readouterr()
            assert "No parameter differences found." in captured.out
            assert "No performance differences found." in captured.out
            assert "Model Hash Different: False" in captured.out
    
    def test_compare_models_version_not_found(self, temp_model_dir, capsys):
        """Test comparison when version not found."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.version1 = "nonexistent1"
        args.version2 = "nonexistent2"
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.compare_models.side_effect = FileNotFoundError("Version not found")
            mock_metadata_class.return_value = mock_metadata
            
            with pytest.raises(SystemExit) as exc_info:
                compare_models(args)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: Version not found" in captured.out


class TestCleanupOldModels:
    """Test cleanup_old_models functionality."""
    
    def create_test_files(self, temp_model_dir, versions):
        """Helper to create test model and metadata files."""
        for version in versions:
            # Create metadata file
            metadata_file = Path(temp_model_dir) / f"metadata_{version}.json"
            metadata = {
                "version": version,
                "created_at": "2024-01-20T14:30:22.123456",
                "model_info": {"file_name": f"{version}.h5"}
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Create model file
            model_file = Path(temp_model_dir) / f"{version}.h5"
            model_file.write_text("dummy model data")
    
    def test_cleanup_nothing_to_clean(self, temp_model_dir, capsys):
        """Test cleanup when there are fewer models than keep threshold."""
        self.create_test_files(temp_model_dir, ["v1", "v2", "v3"])
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 5
        args.force = False
        
        mock_versions = [
            {"version": "v3", "created_at": "2024-01-22T10:00:00", "model_file": "v3.h5"},
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            cleanup_old_models(args)
            
            captured = capsys.readouterr()
            assert "Only 3 versions found, nothing to clean up." in captured.out
    
    def test_cleanup_with_confirmation_yes(self, temp_model_dir, capsys):
        """Test cleanup with user confirmation (yes)."""
        self.create_test_files(temp_model_dir, ["v1", "v2", "v3", "v4", "v5", "v6"])
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 3
        args.force = False
        
        mock_versions = [
            {"version": "v6", "created_at": "2024-01-25T10:00:00", "model_file": "v6.h5"},
            {"version": "v5", "created_at": "2024-01-24T10:00:00", "model_file": "v5.h5"},
            {"version": "v4", "created_at": "2024-01-23T10:00:00", "model_file": "v4.h5"},
            {"version": "v3", "created_at": "2024-01-22T10:00:00", "model_file": "v3.h5"},
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            with patch('builtins.input', return_value='y'):
                cleanup_old_models(args)
                
                captured = capsys.readouterr()
                assert "Will delete 3 old model versions:" in captured.out
                assert "v3" in captured.out
                assert "v2" in captured.out
                assert "v1" in captured.out
                assert "Cleanup complete. Deleted 3 model versions." in captured.out
    
    def test_cleanup_with_confirmation_no(self, temp_model_dir, capsys):
        """Test cleanup with user confirmation (no)."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 3
        args.force = False
        
        mock_versions = [
            {"version": "v6", "created_at": "2024-01-25T10:00:00", "model_file": "v6.h5"},
            {"version": "v5", "created_at": "2024-01-24T10:00:00", "model_file": "v5.h5"},
            {"version": "v4", "created_at": "2024-01-23T10:00:00", "model_file": "v4.h5"},
            {"version": "v3", "created_at": "2024-01-22T10:00:00", "model_file": "v3.h5"},
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            with patch('builtins.input', return_value='n'):
                cleanup_old_models(args)
                
                captured = capsys.readouterr()
                assert "Cleanup cancelled." in captured.out
    
    def test_cleanup_force_mode(self, temp_model_dir, capsys):
        """Test cleanup in force mode (no confirmation)."""
        self.create_test_files(temp_model_dir, ["v1", "v2", "v3"])
        
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 1
        args.force = True
        
        mock_versions = [
            {"version": "v3", "created_at": "2024-01-22T10:00:00", "model_file": "v3.h5"},
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            cleanup_old_models(args)
            
            captured = capsys.readouterr()
            assert "Will delete 2 old model versions:" in captured.out
            assert "Cleanup complete. Deleted 2 model versions." in captured.out
            # Should not show confirmation prompt
            assert "Continue? (y/N):" not in captured.out
    
    def test_cleanup_missing_files_handled(self, temp_model_dir, capsys):
        """Test cleanup handles missing metadata/model files gracefully."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 1
        args.force = True
        
        mock_versions = [
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            cleanup_old_models(args)
            
            captured = capsys.readouterr()
            assert "Cleanup complete. Deleted 1 model versions." in captured.out


class TestMainCLI:
    """Test main CLI entry point."""
    
    def test_main_no_command_prints_help(self):
        """Test main with no command prints help and exits."""
        with patch('sys.argv', ['model_manager']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('src.model_manager.list_models')
    def test_main_list_command(self, mock_list_models):
        """Test main with list command."""
        with patch('sys.argv', ['model_manager', 'list']):
            main()
            mock_list_models.assert_called_once()
    
    @patch('src.model_manager.show_metadata')
    def test_main_show_command(self, mock_show_metadata):
        """Test main with show command."""
        with patch('sys.argv', ['model_manager', 'show', 'v20240120_143022']):
            main()
            mock_show_metadata.assert_called_once()
    
    @patch('src.model_manager.compare_models')
    def test_main_compare_command(self, mock_compare_models):
        """Test main with compare command."""
        with patch('sys.argv', ['model_manager', 'compare', 'v1', 'v2']):
            main()
            mock_compare_models.assert_called_once()
    
    @patch('src.model_manager.cleanup_old_models')
    def test_main_cleanup_command(self, mock_cleanup):
        """Test main with cleanup command."""
        with patch('sys.argv', ['model_manager', 'cleanup', '--keep', '3', '--force']):
            main()
            mock_cleanup.assert_called_once()
    
    def test_main_with_custom_model_directory(self):
        """Test main with custom model directory."""
        with patch('sys.argv', ['model_manager', '--model-directory', '/custom/path', 'list']):
            with patch('src.model_manager.list_models') as mock_list:
                main()
                args = mock_list.call_args[0][0]
                assert args.model_directory == '/custom/path'


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_cleanup_with_permission_error(self, temp_model_dir):
        """Test cleanup handles permission errors gracefully."""
        args = argparse.Namespace()
        args.model_directory = temp_model_dir
        args.keep = 1
        args.force = True
        
        mock_versions = [
            {"version": "v2", "created_at": "2024-01-21T10:00:00", "model_file": "v2.h5"},
            {"version": "v1", "created_at": "2024-01-20T10:00:00", "model_file": "v1.h5"}
        ]
        
        # Create a metadata file that will cause permission error
        metadata_file = Path(temp_model_dir) / "metadata_v1.json"
        metadata_file.write_text('{"version": "v1"}')
        metadata_file.chmod(0o000)  # No permissions
        
        with patch('src.model_manager.ModelMetadata') as mock_metadata_class:
            mock_metadata = MagicMock()
            mock_metadata.list_model_versions.return_value = mock_versions
            mock_metadata_class.return_value = mock_metadata
            
            try:
                # This might raise PermissionError, but should not crash the program
                cleanup_old_models(args)
            except PermissionError:
                pass  # Expected in some environments
            finally:
                # Clean up
                metadata_file.chmod(0o644)


if __name__ == "__main__":
    pytest.main([__file__])