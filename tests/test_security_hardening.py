"""
Test cases for security hardening features.
Tests path sanitization, file validation, and secure operations.
"""
import pytest
import tempfile
import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from security_utils import (
    sanitize_path,
    validate_file_path,
    secure_json_load,
    validate_joblib_file,
    sanitize_error_message,
    validate_file_size
)


class TestPathSanitization:
    """Test path sanitization and validation."""
    
    def test_sanitize_path_removes_dangerous_sequences(self):
        """Test that path sanitization removes dangerous sequences."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/../../etc/shadow",
            "legitimate_file/../../../dangerous_file"
        ]
        
        for dangerous_path in dangerous_paths:
            sanitized = sanitize_path(dangerous_path)
            assert ".." not in sanitized
            assert not os.path.isabs(sanitized)
    
    def test_sanitize_path_preserves_legitimate_paths(self):
        """Test that legitimate paths are preserved."""
        legitimate_paths = [
            "models/autoencoder.h5",
            "data/sensor_data.csv",
            "config/settings.yaml"
        ]
        
        for path in legitimate_paths:
            sanitized = sanitize_path(path)
            assert sanitized == path
    
    def test_validate_file_path_rejects_traversal(self):
        """Test that file path validation rejects traversal attempts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a legitimate file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # Valid path should pass
            assert validate_file_path(test_file, temp_dir) == test_file
            
            # Traversal attempts should fail
            with pytest.raises(ValueError, match="Path traversal"):
                validate_file_path("../../../etc/passwd", temp_dir)
    
    def test_validate_file_path_requires_existing_file(self):
        """Test that file path validation requires existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = os.path.join(temp_dir, "nonexistent.txt")
            
            with pytest.raises(ValueError, match="does not exist"):
                validate_file_path(nonexistent_file, temp_dir)


class TestSecureFileOperations:
    """Test secure file loading operations."""
    
    def test_secure_json_load_size_limit(self):
        """Test that JSON loading enforces size limits."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Create a large JSON file
            large_data = {"key": "x" * 1000000}  # 1MB+ JSON
            json.dump(large_data, f)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="File too large"):
                    secure_json_load(f.name, max_size_mb=0.5)
            finally:
                os.unlink(f.name)
    
    def test_secure_json_load_normal_file(self):
        """Test that normal JSON files load correctly."""
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            f.flush()
            
            try:
                loaded_data = secure_json_load(f.name)
                assert loaded_data == test_data
            finally:
                os.unlink(f.name)
    
    def test_validate_joblib_file_integrity(self):
        """Test joblib file integrity validation."""
        # This would require implementing checksum validation
        # For now, test basic file existence and extension
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            f.write(b"fake joblib content")
            f.flush()
            
            try:
                # Should not raise for existing .joblib file
                result = validate_joblib_file(f.name)
                assert result == f.name
            finally:
                os.unlink(f.name)
    
    def test_validate_joblib_file_wrong_extension(self):
        """Test that joblib validation rejects wrong extensions."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"content")
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="Invalid file extension"):
                    validate_joblib_file(f.name)
            finally:
                os.unlink(f.name)


class TestErrorMessageSanitization:
    """Test error message sanitization."""
    
    def test_sanitize_error_message_removes_paths(self):
        """Test that error messages remove sensitive paths."""
        sensitive_messages = [
            "Failed to load /home/user/secret/file.txt",
            "Error accessing C:\\Users\\admin\\passwords.txt",
            "Cannot read /etc/passwd file"
        ]
        
        for message in sensitive_messages:
            sanitized = sanitize_error_message(message)
            assert "/home/" not in sanitized
            assert "C:\\" not in sanitized
            assert "/etc/" not in sanitized
    
    def test_sanitize_error_message_preserves_safe_content(self):
        """Test that safe error content is preserved."""
        safe_message = "Invalid input format: expected JSON"
        sanitized = sanitize_error_message(safe_message)
        assert "Invalid input format" in sanitized
        assert "JSON" in sanitized


class TestFileSizeValidation:
    """Test file size validation."""
    
    def test_validate_file_size_accepts_small_files(self):
        """Test that small files pass validation."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small content")
            f.flush()
            
            try:
                assert validate_file_size(f.name, max_size_mb=1.0)
            finally:
                os.unlink(f.name)
    
    def test_validate_file_size_rejects_large_files(self):
        """Test that large files are rejected."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * (2 * 1024 * 1024))  # 2MB
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="File too large"):
                    validate_file_size(f.name, max_size_mb=1.0)
            finally:
                os.unlink(f.name)