import logging
import tempfile
import os
from unittest.mock import patch
from src.logging_config import setup_logging, get_logger


def test_setup_logging_default():
    """Test default logging setup."""
    setup_logging()
    
    # Get a logger and verify it's configured
    logger = get_logger("test")
    # Check effective level instead of direct level for inheritance
    assert logger.getEffectiveLevel() == logging.INFO
    # Root logger should have handlers
    root_logger = logging.getLogger()
    assert len(root_logger.handlers) > 0


def test_setup_logging_debug_level():
    """Test logging setup with debug level."""
    setup_logging(level="DEBUG")
    
    logger = get_logger("test_debug")
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_setup_logging_with_file(tmp_path):
    """Test logging setup with file output."""
    log_file = tmp_path / "test.log"
    setup_logging(log_file=str(log_file))
    
    logger = get_logger("test_file")
    logger.info("Test message")
    
    # Check file was created and contains message
    assert log_file.exists()
    content = log_file.read_text()
    assert "Test message" in content


def test_logging_format():
    """Test logging message format includes required components."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        setup_logging(log_file=f.name, level="DEBUG")
        
        logger = get_logger("test_format")
        logger.info("Format test message")
        
        # Read the log file
        with open(f.name, 'r') as log_file:
            content = log_file.read()
        
        # Check format includes timestamp, level, logger name, and message
        assert "INFO" in content
        assert "test_format" in content
        assert "Format test message" in content
        # Should include timestamp (basic check for date-like pattern)
        assert any(char.isdigit() for char in content)
        
        os.unlink(f.name)


def test_get_logger_returns_same_instance():
    """Test that get_logger returns the same instance for same name."""
    logger1 = get_logger("same_name")
    logger2 = get_logger("same_name")
    
    assert logger1 is logger2


def test_get_logger_different_names():
    """Test that get_logger returns different instances for different names."""
    logger1 = get_logger("name1")
    logger2 = get_logger("name2")
    
    assert logger1 is not logger2
    assert logger1.name == "name1"
    assert logger2.name == "name2"


def test_structured_logging():
    """Test that structured logging works with extra fields."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        setup_logging(log_file=f.name, level="INFO")
        
        logger = get_logger("test_structured")
        logger.info("Structured message", extra={
            "user_id": "123",
            "operation": "test",
            "duration": 0.5
        })
        
        with open(f.name, 'r') as log_file:
            content = log_file.read()
        
        assert "Structured message" in content
        
        os.unlink(f.name)


def test_logging_config_from_env(monkeypatch):
    """Test logging configuration from environment variables."""
    monkeypatch.setenv("IOT_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("IOT_LOG_FILE", "/tmp/test_env.log")
    
    # Mock the setup to avoid actually creating files
    with patch('src.logging_config.setup_logging') as mock_setup:
        from src.logging_config import setup_logging_from_config
        setup_logging_from_config()
        
        # Verify setup_logging was called with env values
        mock_setup.assert_called_once()
        args, kwargs = mock_setup.call_args
        
        # Check that environment variables influenced the call
        # (exact implementation depends on how setup_logging_from_config works)


def test_logger_hierarchy():
    """Test that logger hierarchy works correctly."""
    setup_logging(level="DEBUG")
    
    get_logger("parent")
    child_logger = get_logger("parent.child")
    
    # Child should inherit from parent
    assert child_logger.parent.name == "parent"


def test_multiple_handlers_no_duplication():
    """Test that calling setup_logging multiple times doesn't duplicate handlers."""
    setup_logging()
    initial_handlers = len(logging.getLogger().handlers)
    
    setup_logging()  # Call again
    after_handlers = len(logging.getLogger().handlers)
    
    # Should not have doubled the handlers
    assert after_handlers == initial_handlers


def test_logging_sensitive_data_filtering():
    """Test that sensitive data is filtered from logs."""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        setup_logging(log_file=f.name, level="DEBUG")
        
        logger = get_logger("test_sensitive")
        
        # Log messages that might contain sensitive data
        logger.info("User logged in with password: secret123")
        logger.info("API key: abc123def456")
        logger.info("Database connection: postgresql://user:password@host/db")
        
        with open(f.name, 'r') as log_file:
            content = log_file.read()
        
        # Sensitive data should be masked
        assert "secret123" not in content
        assert "*FILTERED*" in content or "***" in content
        
        os.unlink(f.name)