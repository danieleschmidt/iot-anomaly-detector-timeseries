"""
Security tests for the IoT anomaly detection system.
"""

import pytest
import tempfile
import os
import stat
from pathlib import Path
from unittest.mock import patch, mock_open

from src.security_utils import SecurityUtils
from src.config import Config


@pytest.mark.security
class TestSecurityHardening:
    """Test security features and hardening measures."""
    
    def test_file_permissions(self, temp_dir):
        """Test that sensitive files have appropriate permissions."""
        
        # Create test files
        model_file = temp_dir / "test_model.h5"
        config_file = temp_dir / "config.yaml"
        key_file = temp_dir / "secret.key"
        
        model_file.touch()
        config_file.touch()
        key_file.touch()
        
        # Test setting secure permissions
        security_utils = SecurityUtils()
        
        # Secure model file
        security_utils.secure_file_permissions(str(model_file))
        model_perms = oct(model_file.stat().st_mode)[-3:]
        assert model_perms in ["600", "640"]  # Owner read/write only or group read
        
        # Secure config file
        security_utils.secure_file_permissions(str(config_file))
        config_perms = oct(config_file.stat().st_mode)[-3:]
        assert config_perms in ["600", "640"]
        
        # Secure key file (most restrictive)
        security_utils.secure_file_permissions(str(key_file), mode=0o600)
        key_perms = oct(key_file.stat().st_mode)[-3:]
        assert key_perms == "600"
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        
        security_utils = SecurityUtils()
        
        # Test file path validation
        valid_paths = [
            "/tmp/model.h5",
            "data/sensor_data.csv",
            "../data/test.csv"
        ]
        
        invalid_paths = [
            "/etc/passwd",
            "../../etc/shadow",
            "/dev/null",
            "",
            None
        ]
        
        for path in valid_paths:
            assert security_utils.validate_file_path(path, allowed_extensions=[".h5", ".csv"])
        
        for path in invalid_paths:
            assert not security_utils.validate_file_path(path, allowed_extensions=[".h5", ".csv"])
        
        # Test parameter validation
        valid_params = {
            "window_size": 30,
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        invalid_params = {
            "window_size": -1,
            "epochs": 0,
            "batch_size": "invalid",
            "learning_rate": 10.0  # Too high
        }
        
        assert security_utils.validate_training_params(valid_params)
        assert not security_utils.validate_training_params(invalid_params)
    
    def test_data_sanitization(self):
        """Test data sanitization for potential injection attacks."""
        
        security_utils = SecurityUtils()
        
        # Test SQL injection patterns in data
        malicious_data = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "`rm -rf /`"
        ]
        
        for data in malicious_data:
            sanitized = security_utils.sanitize_string_input(data)
            
            # Should remove or escape dangerous characters
            assert "'" not in sanitized or "\\'" in sanitized
            assert "<script>" not in sanitized
            assert "`" not in sanitized
            assert "$(" not in sanitized
    
    def test_configuration_security(self, temp_dir):
        """Test secure configuration handling."""
        
        # Create test config with sensitive data
        config_content = """
        database:
          password: secret123
          connection_string: postgresql://user:pass@localhost:5432/db
        
        api:
          secret_key: super-secret-key
          jwt_secret: jwt-signing-key
        
        external_services:
          api_key: sk-1234567890abcdef
        """
        
        config_file = temp_dir / "test_config.yaml"
        config_file.write_text(config_content)
        
        # Test that config loader masks sensitive fields
        config = Config(config_path=str(config_file))
        
        # Sensitive fields should be masked in logs/exports
        masked_config = config.get_masked_config()
        
        assert "***" in str(masked_config) or masked_config is None
        
        # Original config should still work for actual usage
        # (This would be implemented in the Config class)
    
    def test_model_integrity(self, temp_dir):
        """Test model file integrity checks."""
        
        security_utils = SecurityUtils()
        
        # Create test model file
        model_file = temp_dir / "test_model.h5"
        model_content = b"fake model content for testing"
        model_file.write_bytes(model_content)
        
        # Calculate checksum
        original_checksum = security_utils.calculate_file_checksum(str(model_file))
        assert original_checksum is not None
        assert len(original_checksum) == 64  # SHA256 hex length
        
        # Verify integrity
        assert security_utils.verify_file_integrity(str(model_file), original_checksum)
        
        # Modify file and test integrity failure
        model_file.write_bytes(b"modified content")
        assert not security_utils.verify_file_integrity(str(model_file), original_checksum)
    
    def test_secure_temporary_files(self, temp_dir):
        """Test secure handling of temporary files."""
        
        security_utils = SecurityUtils()
        
        # Create secure temp file
        with security_utils.secure_temp_file(suffix=".csv") as temp_file:
            temp_path = Path(temp_file.name)
            
            # Verify file exists and has secure permissions
            assert temp_path.exists()
            perms = oct(temp_path.stat().st_mode)[-3:]
            assert perms == "600"  # Owner only
            
            # Write some data
            temp_file.write(b"test,data,here")
            temp_file.flush()
            
            # Verify data was written
            assert temp_path.stat().st_size > 0
        
        # File should be automatically deleted
        assert not temp_path.exists()
    
    def test_secrets_detection(self):
        """Test detection of potential secrets in code/config."""
        
        security_utils = SecurityUtils()
        
        # Test strings that look like secrets
        potential_secrets = [
            "api_key = 'sk-1234567890abcdef'",
            "password: secretpassword123",
            "token = 'ghp_1234567890abcdef'",
            "aws_secret_access_key = 'AKIA1234567890ABCDEF'",
            "private_key = '-----BEGIN PRIVATE KEY-----'"
        ]
        
        safe_strings = [
            "api_key = os.environ.get('API_KEY')",
            "password: '***'",
            "token = get_token_from_vault()",
            "# This is just a comment about api_key",
            "test_data = 'sample123'"
        ]
        
        for secret in potential_secrets:
            assert security_utils.detect_potential_secret(secret)
        
        for safe in safe_strings:
            assert not security_utils.detect_potential_secret(safe)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        
        security_utils = SecurityUtils()
        
        # Test API rate limiting
        client_id = "test_client"
        
        # Should allow initial requests
        for _ in range(5):
            assert security_utils.check_rate_limit(client_id, limit=10, window=60)
        
        # Should start limiting after threshold
        for _ in range(10):  # Exceed limit
            security_utils.check_rate_limit(client_id, limit=10, window=60)
        
        # Should be rate limited now
        assert not security_utils.check_rate_limit(client_id, limit=10, window=60)
    
    def test_audit_logging(self, temp_dir):
        """Test security audit logging."""
        
        security_utils = SecurityUtils()
        log_file = temp_dir / "audit.log"
        
        # Configure audit logging
        security_utils.setup_audit_logging(str(log_file))
        
        # Log security events
        security_utils.audit_log("user_login", {"user": "test_user", "ip": "127.0.0.1"})
        security_utils.audit_log("model_access", {"model": "autoencoder_v1", "action": "load"})
        security_utils.audit_log("config_change", {"setting": "api_key", "action": "update"})
        
        # Verify logs were written
        assert log_file.exists()
        log_content = log_file.read_text()
        
        assert "user_login" in log_content
        assert "model_access" in log_content
        assert "config_change" in log_content
        assert "test_user" in log_content
    
    @pytest.mark.slow
    def test_vulnerability_scanning(self):
        """Test for known vulnerability patterns."""
        
        security_utils = SecurityUtils()
        
        # Test code patterns that might be vulnerable
        vulnerable_patterns = [
            "eval(user_input)",
            "exec(data)",
            "pickle.loads(untrusted_data)",
            "subprocess.call(user_command, shell=True)",
            "os.system(user_input)"
        ]
        
        safe_patterns = [
            "eval('2 + 2')",  # Static eval
            "pickle.loads(trusted_data)",
            "subprocess.call(['ls', '-l'])",  # List form
            "os.system('ls')"  # Static command
        ]
        
        for pattern in vulnerable_patterns:
            vulnerabilities = security_utils.scan_code_vulnerabilities(pattern)
            assert len(vulnerabilities) > 0
        
        for pattern in safe_patterns:
            vulnerabilities = security_utils.scan_code_vulnerabilities(pattern)
            # Some might still be flagged, but fewer issues
            assert len(vulnerabilities) <= 1
    
    def test_environment_security(self):
        """Test environment security configurations."""
        
        security_utils = SecurityUtils()
        
        # Test secure environment setup
        env_vars = {
            'DEBUG': 'False',
            'SECRET_KEY': 'not-default-key',
            'DATABASE_URL': 'postgresql://...',
            'ALLOWED_HOSTS': 'localhost,127.0.0.1'
        }
        
        with patch.dict(os.environ, env_vars):
            security_check = security_utils.validate_environment_security()
            
            # Should pass basic security checks
            assert security_check['debug_disabled']
            assert security_check['secret_key_set']
            assert security_check['allowed_hosts_configured']
        
        # Test insecure environment
        insecure_env = {
            'DEBUG': 'True',
            'SECRET_KEY': 'default',
            'ALLOWED_HOSTS': '*'
        }
        
        with patch.dict(os.environ, insecure_env, clear=True):
            security_check = security_utils.validate_environment_security()
            
            # Should fail security checks
            assert not security_check['debug_disabled']
            assert not security_check['secret_key_set']


@pytest.mark.security  
class TestDataPrivacy:
    """Test data privacy and compliance features."""
    
    def test_data_anonymization(self):
        """Test data anonymization capabilities."""
        
        security_utils = SecurityUtils()
        
        # Test PII detection and anonymization
        sensitive_data = {
            'email': 'user@example.com',
            'phone': '+1-555-123-4567',
            'ssn': '123-45-6789',
            'credit_card': '4111-1111-1111-1111',
            'ip_address': '192.168.1.100'
        }
        
        anonymized = security_utils.anonymize_sensitive_data(sensitive_data)
        
        # Should not contain original sensitive values
        for key, value in sensitive_data.items():
            assert value not in str(anonymized.values())
        
        # Should contain anonymized placeholders or hashed values
        assert 'email' in anonymized
        assert '@' not in anonymized['email'] or '***' in anonymized['email']
    
    def test_data_retention_policy(self, temp_dir):
        """Test data retention and cleanup policies."""
        
        security_utils = SecurityUtils()
        
        # Create old files
        old_file = temp_dir / "old_data.csv"
        recent_file = temp_dir / "recent_data.csv"
        
        old_file.touch()
        recent_file.touch()
        
        # Simulate old file (modify timestamp)
        old_timestamp = 1609459200  # Jan 1, 2021
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Apply retention policy (delete files older than 1 year)
        deleted_files = security_utils.apply_retention_policy(
            str(temp_dir), 
            max_age_days=365,
            file_patterns=["*.csv"]
        )
        
        # Old file should be marked for deletion
        assert str(old_file) in deleted_files
        assert str(recent_file) not in deleted_files
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        
        security_utils = SecurityUtils()
        
        # Test right to be forgotten
        user_data = {
            'user_id': '12345',
            'sensor_data': [1.0, 2.0, 3.0, 4.0],
            'metadata': {'location': 'EU', 'consent': True}
        }
        
        # Test data export (right to data portability)
        exported_data = security_utils.export_user_data('12345', user_data)
        assert 'user_id' in exported_data
        assert 'sensor_data' in exported_data
        assert isinstance(exported_data, dict)
        
        # Test data deletion (right to be forgotten)
        anonymized_data = security_utils.anonymize_user_data('12345', user_data)
        assert anonymized_data['user_id'] != '12345'
        assert 'sensor_data' in anonymized_data  # Data preserved but anonymized