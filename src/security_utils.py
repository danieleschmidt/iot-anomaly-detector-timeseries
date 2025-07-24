"""
Security utilities for input validation, path sanitization, and secure file operations.
Provides protection against path traversal, injection attacks, and unsafe file operations.
"""
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def sanitize_path(file_path: str) -> str:
    """
    Sanitize file path to prevent directory traversal attacks.
    
    Args:
        file_path: The file path to sanitize
        
    Returns:
        Sanitized file path with dangerous sequences removed
        
    Example:
        >>> sanitize_path("../../../etc/passwd")
        'etc/passwd'
        >>> sanitize_path("legitimate/file.txt")
        'legitimate/file.txt'
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    
    # Convert to Path object and resolve to remove .. sequences
    path = Path(file_path)
    
    # Check if original was absolute
    was_absolute = path.is_absolute()
    
    # Remove any parent directory references but preserve legitimate parts
    parts = []
    for part in path.parts:
        if part == '..' or part == '.':
            continue  # Skip traversal attempts
        if part.startswith('.') and part not in {'.gitignore', '.env'}:
            continue  # Skip hidden files except common legitimate ones
        parts.append(part)
    
    # Reconstruct path without dangerous components
    if not parts:
        return ''
        
    # Preserve absolute path nature for legitimate absolute paths
    if was_absolute and parts[0] != '/':
        sanitized = str(Path('/', *parts))
    else:
        sanitized = str(Path(*parts))
    
    return sanitized


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> str:
    """
    Validate file path for security and existence.
    
    Args:
        file_path: The file path to validate
        base_dir: Optional base directory to constrain file access
        
    Returns:
        Validated file path
        
    Raises:
        ValueError: If path is unsafe or file doesn't exist
    """
    # Sanitize the path first
    sanitized_path = sanitize_path(file_path)
    
    # If base_dir is provided, ensure path is within it
    if base_dir:
        base_path = Path(base_dir).resolve()
        full_path = (base_path / sanitized_path).resolve()
        
        # Check if resolved path is within base directory
        try:
            full_path.relative_to(base_path)
        except ValueError:
            raise ValueError(f"Path traversal attempt detected: {file_path}")
        
        file_path = str(full_path)
    else:
        file_path = sanitized_path
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    return file_path


def secure_json_load(file_path: str, max_size_mb: float = 10.0) -> Any:
    """
    Securely load JSON file with size limits.
    
    Args:
        file_path: Path to JSON file
        max_size_mb: Maximum file size in MB
        
    Returns:
        Parsed JSON data
        
    Raises:
        ValueError: If file is too large or invalid JSON
    """
    # Validate file size first
    validate_file_size(file_path, max_size_mb)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {sanitize_error_message(str(e))}") from e
    except Exception as e:
        raise ValueError(f"Failed to load JSON: {sanitize_error_message(str(e))}") from e


def validate_joblib_file(file_path: str) -> str:
    """
    Validate joblib file for safe loading.
    
    Args:
        file_path: Path to joblib file
        
    Returns:
        Validated file path
        
    Raises:
        ValueError: If file is unsafe to load
    """
    # Check file extension
    if not file_path.lower().endswith(('.joblib', '.pkl', '.pickle')):
        raise ValueError(f"Invalid file extension for joblib file: {file_path}")
    
    # Validate path
    validated_path = validate_file_path(file_path)
    
    # Additional integrity checks could be added here
    # For example: checksum validation, digital signatures
    
    logger.info(f"Validated joblib file: {os.path.basename(validated_path)}")
    return validated_path


def sanitize_error_message(error_msg: str) -> str:
    """
    Sanitize error messages to remove sensitive information.
    
    Args:
        error_msg: Original error message
        
    Returns:
        Sanitized error message
    """
    if not isinstance(error_msg, str):
        error_msg = str(error_msg)
    
    # Patterns to remove/replace
    patterns = [
        # Remove absolute paths
        (re.compile(r'/[^\s]*(?:/[^\s]*)+'), '[PATH_REMOVED]'),
        (re.compile(r'[A-Z]:\\[^\s]*(?:\\[^\s]*)+'), '[PATH_REMOVED]'),
        # Remove usernames
        (re.compile(r'/home/[^/\s]+'), '[USER_HOME]'),
        (re.compile(r'\\Users\\[^\\s]+'), '[USER_HOME]'),
        # Remove potential secrets (basic patterns)
        (re.compile(r'password[=:\s]+[^\s]+', re.IGNORECASE), 'password=[REDACTED]'),
        (re.compile(r'token[=:\s]+[^\s]+', re.IGNORECASE), 'token=[REDACTED]'),
        (re.compile(r'key[=:\s]+[^\s]+', re.IGNORECASE), 'key=[REDACTED]'),
    ]
    
    sanitized = error_msg
    for pattern, replacement in patterns:
        sanitized = pattern.sub(replacement, sanitized)
    
    return sanitized


def validate_file_size(file_path: str, max_size_mb: float) -> bool:
    """
    Validate file size against maximum limit.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file size is acceptable
        
    Raises:
        ValueError: If file is too large
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    
    file_size = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        raise ValueError(
            f"File too large: {size_mb:.2f}MB exceeds limit of {max_size_mb}MB"
        )
    
    return True


def create_file_checksum(file_path: str) -> str:
    """
    Create SHA-256 checksum for file integrity verification.
    
    Args:
        file_path: Path to file
        
    Returns:
        SHA-256 hexdigest of file content
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def verify_file_checksum(file_path: str, expected_checksum: str) -> bool:
    """
    Verify file integrity using SHA-256 checksum.
    
    Args:
        file_path: Path to file
        expected_checksum: Expected SHA-256 checksum
        
    Returns:
        True if checksum matches, False otherwise
    """
    actual_checksum = create_file_checksum(file_path)
    return actual_checksum.lower() == expected_checksum.lower()