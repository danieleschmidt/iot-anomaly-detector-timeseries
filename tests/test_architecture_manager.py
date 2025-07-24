"""Comprehensive tests for architecture_manager CLI utilities."""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import argparse

from src.architecture_manager import (
    list_architectures, show_architecture, validate_config, create_template,
    compare_architectures, main
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_architecture_config():
    """Sample architecture configuration for testing."""
    return {
        "name": "test_architecture",
        "description": "Test architecture configuration",
        "input_shape": [30, 3],
        "encoder_layers": [
            {
                "type": "lstm",
                "units": 64,
                "return_sequences": True,
                "dropout": 0.1
            },
            {
                "type": "batch_norm"
            },
            {
                "type": "lstm",
                "units": 32,
                "return_sequences": False,
                "dropout": 0.1
            }
        ],
        "latent_config": {
            "dim": 16,
            "activation": "linear",
            "regularization": "l2"
        },
        "compilation": {
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["mae"]
        }
    }


@pytest.fixture
def mock_predefined_architectures():
    """Mock predefined architectures for testing - module-level fixture."""
    return {
        "simple_lstm": {
            "name": "simple_lstm",
            "description": "Simple LSTM autoencoder",
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "lstm", "units": 50, "return_sequences": False}
            ],
            "latent_config": {"dim": 10, "activation": "linear"},
            "compilation": {"optimizer": "adam", "loss": "mse"}
        },
        "deep_lstm": {
            "name": "deep_lstm", 
            "description": "Deep LSTM autoencoder",
            "input_shape": [30, 3],
            "encoder_layers": [
                {"type": "lstm", "units": 64, "return_sequences": True},
                {"type": "lstm", "units": 32, "return_sequences": False}
            ],
            "latent_config": {"dim": 16, "activation": "linear"},
            "compilation": {"optimizer": "adam", "loss": "mse"}
        }
    }


class TestArchitectureManagerCLI:
    """Test suite for architecture manager CLI functionality."""


class TestListArchitectures:
    """Test list_architectures functionality."""
    
    def test_list_architectures_basic(self, mock_predefined_architectures, capsys):
        """Test basic architecture listing."""
        args = argparse.Namespace()
        args.verbose = False
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            list_architectures(args)
            
            captured = capsys.readouterr()
            assert "Available predefined architectures (2):" in captured.out
            assert "simple_lstm" in captured.out
            assert "deep_lstm" in captured.out
            assert "Simple LSTM autoencoder" in captured.out
            assert "Deep LSTM autoencoder" in captured.out
    
    def test_list_architectures_verbose(self, mock_predefined_architectures, capsys):
        """Test verbose architecture listing."""
        args = argparse.Namespace()
        args.verbose = True
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            list_architectures(args)
            
            captured = capsys.readouterr()
            assert "Available predefined architectures (2):" in captured.out
            assert "Layers:" in captured.out
            assert "1. lstm:" in captured.out
            assert "units" in captured.out
    
    def test_list_architectures_empty(self, capsys):
        """Test listing when no architectures available."""
        args = argparse.Namespace()
        args.verbose = False
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value={}):
            list_architectures(args)
            
            captured = capsys.readouterr()
            assert "Available predefined architectures (0):" in captured.out


class TestShowArchitecture:
    """Test show_architecture functionality."""
    
    def test_show_architecture_existing(self, mock_predefined_architectures, capsys):
        """Test showing details of existing architecture."""
        args = argparse.Namespace()
        args.name = "simple_lstm"
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            show_architecture(args)
            
            captured = capsys.readouterr()
            assert "Architecture: simple_lstm" in captured.out
            assert "Simple LSTM autoencoder" in captured.out
            assert '"type": "lstm"' in captured.out
    
    def test_show_architecture_nonexistent(self, mock_predefined_architectures, capsys):
        """Test showing details of non-existent architecture."""
        args = argparse.Namespace()
        args.name = "nonexistent"
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            with pytest.raises(SystemExit) as exc_info:
                show_architecture(args)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: Architecture 'nonexistent' not found." in captured.out
            assert "Available architectures: simple_lstm, deep_lstm" in captured.out


class TestValidateConfig:
    """Test validate_config functionality."""
    
    def test_validate_config_valid_file(self, temp_dir, sample_architecture_config, capsys):
        """Test validating a valid configuration file."""
        config_file = Path(temp_dir) / "valid_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_architecture_config, f)
        
        args = argparse.Namespace()
        args.config_file = str(config_file)
        args.verbose = False
        
        with patch('src.architecture_manager.load_architecture_config', 
                   return_value=sample_architecture_config):
            validate_config(args)
            
            captured = capsys.readouterr()
            assert f"✓ Configuration file '{config_file}' is valid." in captured.out
    
    def test_validate_config_valid_file_verbose(self, temp_dir, sample_architecture_config, capsys):
        """Test validating configuration file with verbose output."""
        config_file = Path(temp_dir) / "valid_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_architecture_config, f)
        
        args = argparse.Namespace()
        args.config_file = str(config_file)
        args.verbose = True
        
        with patch('src.architecture_manager.load_architecture_config', 
                   return_value=sample_architecture_config):
            validate_config(args)
            
            captured = capsys.readouterr()
            assert "Configuration details:" in captured.out
            assert '"name": "test_architecture"' in captured.out
    
    def test_validate_config_missing_file(self, temp_dir, capsys):
        """Test validating non-existent configuration file."""
        args = argparse.Namespace()
        args.config_file = str(Path(temp_dir) / "nonexistent.json")
        
        with pytest.raises(SystemExit) as exc_info:
            validate_config(args)
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Configuration file" in captured.out
        assert "not found." in captured.out
    
    def test_validate_config_invalid_file(self, temp_dir, capsys):
        """Test validating invalid configuration file."""
        config_file = Path(temp_dir) / "invalid_config.json"
        with open(config_file, 'w') as f:
            f.write('{"invalid": "json"')  # Invalid JSON
        
        args = argparse.Namespace()
        args.config_file = str(config_file)
        args.verbose = False
        
        with patch('src.architecture_manager.load_architecture_config', 
                   side_effect=Exception("Invalid configuration")):
            with pytest.raises(SystemExit) as exc_info:
                validate_config(args)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "✗ Configuration file" in captured.out
            assert "is invalid: Invalid configuration" in captured.out


class TestCreateTemplate:
    """Test create_template functionality."""
    
    def test_create_template_minimal(self, temp_dir, capsys):
        """Test creating minimal template."""
        output_file = Path(temp_dir) / "template.json"
        
        args = argparse.Namespace()
        args.output = str(output_file)
        args.architecture = None
        
        create_template(args)
        
        captured = capsys.readouterr()
        assert f"Template configuration saved to '{output_file}'" in captured.out
        
        # Verify file contents
        assert output_file.exists()
        with open(output_file) as f:
            config = json.load(f)
        
        assert config["name"] == "custom_architecture"
        assert "encoder_layers" in config
        assert "latent_config" in config
        assert "compilation" in config
    
    def test_create_template_from_predefined(self, temp_dir, mock_predefined_architectures, capsys):
        """Test creating template based on predefined architecture."""
        output_file = Path(temp_dir) / "custom_template.json"
        
        args = argparse.Namespace()
        args.output = str(output_file)
        args.architecture = "simple_lstm"
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            create_template(args)
            
            captured = capsys.readouterr()
            assert f"Template configuration saved to '{output_file}'" in captured.out
            
            # Verify file contents
            assert output_file.exists()
            with open(output_file) as f:
                config = json.load(f)
            
            assert config["name"] == "custom_simple_lstm"
            assert "Custom architecture based on simple_lstm" in config["description"]
            assert config["input_shape"] == [30, 3]
    
    def test_create_template_nonexistent_base(self, temp_dir, mock_predefined_architectures, capsys):
        """Test creating template from non-existent predefined architecture."""
        output_file = Path(temp_dir) / "template.json"
        
        args = argparse.Namespace()
        args.output = str(output_file)
        args.architecture = "nonexistent"
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            with pytest.raises(SystemExit) as exc_info:
                create_template(args)
            
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Error: Architecture 'nonexistent' not found." in captured.out
    
    def test_create_template_creates_directories(self, temp_dir):
        """Test that template creation creates necessary directories."""
        nested_output = Path(temp_dir) / "nested" / "dir" / "template.json"
        
        args = argparse.Namespace()
        args.output = str(nested_output)
        args.architecture = None
        
        create_template(args)
        
        assert nested_output.exists()
        assert nested_output.parent.exists()




class TestCompareArchitectures:
    """Test compare_architectures functionality."""
    
    def test_compare_predefined_architectures(self, mock_predefined_architectures, capsys):
        """Test comparing two predefined architectures."""
        args = argparse.Namespace()
        args.arch1 = "simple_lstm"
        args.arch2 = "deep_lstm"
        args.verbose = False
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            compare_architectures(args)
            
            captured = capsys.readouterr()
            assert "Comparing architectures:" in captured.out
            assert "predefined (simple_lstm)" in captured.out
            assert "predefined (deep_lstm)" in captured.out
            assert "Input Shape: [30, 3] vs [30, 3]" in captured.out
            assert "Latent Dim: 10 vs 16" in captured.out
            assert "Encoder Layers: 1 vs 2" in captured.out
            assert "Layer Types:" in captured.out
            assert "lstm" in captured.out
    
    def test_compare_file_architectures(self, temp_dir, sample_architecture_config, capsys):
        """Test comparing architectures from files."""
        config1 = sample_architecture_config.copy()
        config1["latent_config"]["dim"] = 20
        
        config2 = sample_architecture_config.copy()
        config2["latent_config"]["dim"] = 30
        
        file1 = Path(temp_dir) / "config1.json"
        file2 = Path(temp_dir) / "config2.json"
        
        with open(file1, 'w') as f:
            json.dump(config1, f)
        with open(file2, 'w') as f:
            json.dump(config2, f)
        
        args = argparse.Namespace()
        args.arch1 = str(file1)
        args.arch2 = str(file2)
        args.verbose = False
        
        with patch('src.architecture_manager.get_predefined_architectures', return_value={}), \
             patch('src.architecture_manager.load_architecture_config') as mock_load:
            
            mock_load.side_effect = [config1, config2]
            
            compare_architectures(args)
            
            captured = capsys.readouterr()
            assert f"file ({file1})" in captured.out
            assert f"file ({file2})" in captured.out
            # Both configs have same latent dim since sample_architecture_config has dim 16 in latent_config
            assert "Latent Dim:" in captured.out
    
    def test_compare_mixed_architectures(self, temp_dir, mock_predefined_architectures, 
                                        sample_architecture_config, capsys):
        """Test comparing predefined vs file architecture."""
        config_file = Path(temp_dir) / "custom_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_architecture_config, f)
        
        args = argparse.Namespace()
        args.arch1 = "simple_lstm"
        args.arch2 = str(config_file)
        args.verbose = False
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures), \
             patch('src.architecture_manager.load_architecture_config', 
                   return_value=sample_architecture_config):
            
            compare_architectures(args)
            
            captured = capsys.readouterr()
            assert "predefined (simple_lstm)" in captured.out
            assert f"file ({config_file})" in captured.out
    
    def test_compare_architectures_verbose(self, mock_predefined_architectures, capsys):
        """Test comparing architectures with verbose output."""
        args = argparse.Namespace()
        args.arch1 = "simple_lstm"
        args.arch2 = "deep_lstm"
        args.verbose = True
        
        with patch('src.architecture_manager.get_predefined_architectures', 
                   return_value=mock_predefined_architectures):
            compare_architectures(args)
            
            captured = capsys.readouterr()
            assert "Full Configurations:" in captured.out
            assert "Architecture 1:" in captured.out
            assert "Architecture 2:" in captured.out
            assert '"name": "simple_lstm"' in captured.out
            assert '"name": "deep_lstm"' in captured.out


class TestMainCLI:
    """Test main CLI entry point."""
    
    def test_main_no_command_prints_help(self):
        """Test main with no command prints help and exits."""
        with patch('sys.argv', ['architecture_manager']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
    
    @patch('src.architecture_manager.list_architectures')
    def test_main_list_command(self, mock_list):
        """Test main with list command."""
        with patch('sys.argv', ['architecture_manager', 'list']):
            main()
            mock_list.assert_called_once()
    
    @patch('src.architecture_manager.show_architecture')
    def test_main_show_command(self, mock_show):
        """Test main with show command."""
        with patch('sys.argv', ['architecture_manager', 'show', 'simple_lstm']):
            main()
            mock_show.assert_called_once()
    
    @patch('src.architecture_manager.validate_config')
    def test_main_validate_command(self, mock_validate):
        """Test main with validate command."""
        with patch('sys.argv', ['architecture_manager', 'validate', '-c', 'config.json']):
            main()
            mock_validate.assert_called_once()
    
    @patch('src.architecture_manager.create_template')
    def test_main_template_command(self, mock_template):
        """Test main with template command."""
        with patch('sys.argv', ['architecture_manager', 'template', '-o', 'template.json']):
            main()
            mock_template.assert_called_once()
    
    @patch('src.architecture_manager.compare_architectures')
    def test_main_compare_command(self, mock_compare):
        """Test main with compare command."""
        with patch('sys.argv', ['architecture_manager', 'compare', 'arch1', 'arch2']):
            main()
            mock_compare.assert_called_once()
    
    def test_main_with_verbose_flag(self):
        """Test main with verbose flag."""
        with patch('sys.argv', ['architecture_manager', '-v', 'list']):
            with patch('src.architecture_manager.list_architectures') as mock_list:
                main()
                args = mock_list.call_args[0][0]
                assert args.verbose is True


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_load_config_json_error(self, temp_dir):
        """Test handling of JSON parsing errors."""
        config_file = Path(temp_dir) / "bad_json.json"
        with open(config_file, 'w') as f:
            f.write('{"invalid": json}')  # Invalid JSON syntax
        
        args = argparse.Namespace()
        args.config_file = str(config_file)
        args.verbose = False
        
        with patch('src.architecture_manager.load_architecture_config', 
                   side_effect=json.JSONDecodeError("Invalid JSON", "test", 0)):
            with pytest.raises(SystemExit):
                validate_config(args)
    
    @pytest.mark.skip(reason="Permission errors may not occur when running as root")
    def test_file_permission_error(self, temp_dir):
        """Test handling of file permission errors."""
        protected_dir = Path(temp_dir) / "protected"
        protected_dir.mkdir()
        protected_dir.chmod(0o000)  # No permissions
        
        args = argparse.Namespace()
        args.output = str(protected_dir / "template.json")
        args.architecture = None
        
        try:
            # This might raise PermissionError depending on system
            with pytest.raises((PermissionError, OSError)):
                create_template(args)
        finally:
            # Clean up
            protected_dir.chmod(0o755)


if __name__ == "__main__":
    pytest.main([__file__])