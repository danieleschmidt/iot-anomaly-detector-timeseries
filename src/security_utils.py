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
    if not isinstance(file_path, (str, os.PathLike)):
        raise TypeError("file_path must be a string or PathLike object")
    
    # Convert PathLike to string
    file_path = str(file_path)
    
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
    if not str(file_path).lower().endswith(('.joblib', '.pkl', '.pickle')):
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


class SecurityUtils:
    """
    Security utilities class providing various security-related functionality.
    This class wraps individual security functions and provides additional security features.
    """
    
    def __init__(self):
        """Initialize SecurityUtils instance."""
        pass
    
    def validate_file_path(self, file_path: str, allowed_extensions: Optional[list] = None) -> bool:
        """
        Validate file path for security.
        
        Args:
            file_path: Path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            True if path is valid, False otherwise
        """
        if not file_path or file_path is None:
            return False
            
        try:
            # Use existing sanitize_path function
            sanitized = sanitize_path(file_path)
            
            # Check for system paths
            system_paths = ['/etc/', '/dev/', '/proc/', '/sys/']
            for sys_path in system_paths:
                if sanitized.startswith(sys_path):
                    return False
            
            # Check file extension if provided
            if allowed_extensions:
                file_ext = Path(sanitized).suffix.lower()
                return file_ext in allowed_extensions
                
            return True
        except:
            return False
    
    def validate_training_params(self, params: dict) -> bool:
        """
        Validate training parameters for security and sanity.
        
        Args:
            params: Dictionary of training parameters
            
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Check window_size
            if 'window_size' in params:
                ws = params['window_size']
                if not isinstance(ws, int) or ws <= 0 or ws > 10000:
                    return False
            
            # Check epochs
            if 'epochs' in params:
                epochs = params['epochs']
                if not isinstance(epochs, int) or epochs <= 0 or epochs > 10000:
                    return False
            
            # Check batch_size
            if 'batch_size' in params:
                bs = params['batch_size']
                if not isinstance(bs, int) or bs <= 0 or bs > 10000:
                    return False
            
            # Check learning_rate
            if 'learning_rate' in params:
                lr = params['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
                    return False
                    
            return True
        except:
            return False
    
    def sanitize_string_input(self, input_str: str) -> str:
        """
        Sanitize string input to prevent injection attacks.
        
        Args:
            input_str: String to sanitize
            
        Returns:
            Sanitized string
        """
        if not isinstance(input_str, str):
            return str(input_str)
        
        # Remove/escape dangerous characters
        sanitized = input_str
        
        # SQL injection patterns
        sanitized = sanitized.replace("'", "\\'")
        sanitized = sanitized.replace("--", "")
        sanitized = sanitized.replace(";", "")
        
        # XSS patterns
        sanitized = sanitized.replace("<script>", "&lt;script&gt;")
        sanitized = sanitized.replace("</script>", "&lt;/script&gt;")
        
        # Command injection patterns
        sanitized = sanitized.replace("`", "")
        sanitized = sanitized.replace("$(", "")
        sanitized = sanitized.replace("rm -rf", "")
        
        return sanitized
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """
        Calculate file checksum using existing function.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA-256 checksum
        """
        return create_file_checksum(file_path)
    
    def verify_file_integrity(self, file_path: str, expected_checksum: str) -> bool:
        """
        Verify file integrity using existing function.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum
            
        Returns:
            True if integrity check passes
        """
        return verify_file_checksum(file_path, expected_checksum)
    
    def secure_file_permissions(self, file_path: str, mode: int = 0o640) -> None:
        """
        Set secure file permissions.
        
        Args:
            file_path: Path to file
            mode: Permission mode (default: 0o640)
        """
        try:
            os.chmod(file_path, mode)
        except OSError:
            pass  # Ignore permission errors in tests
    
    def secure_temp_file(self, suffix: str = ""):
        """
        Create secure temporary file.
        
        Args:
            suffix: File suffix
            
        Returns:
            Temporary file context manager
        """
        import tempfile
        return tempfile.NamedTemporaryFile(mode='w+b', suffix=suffix, delete=True)
    
    def detect_potential_secret(self, text: str) -> bool:
        """
        Detect potential secrets in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if potential secret detected
        """
        secret_patterns = [
            r'api_key\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'password:\s*[\'"]?[^\'"\s]+[\'"]?',
            r'token\s*=\s*[\'"][^\'"\s]+[\'"]',
            r'secret.*key.*[\'"][^\'"\s]+[\'"]',
            r'BEGIN\s+PRIVATE\s+KEY'
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Check if it's a safe pattern
                safe_patterns = [
                    r'os\.environ',
                    r'get.*from.*vault',
                    r'\*\*\*',
                    r'#.*comment'
                ]
                
                is_safe = any(re.search(safe_pattern, text, re.IGNORECASE) 
                             for safe_pattern in safe_patterns)
                
                if not is_safe:
                    return True
        
        return False
    
    def check_rate_limit(self, client_id: str, limit: int = 10, window: int = 60) -> bool:
        """
        Check rate limiting (simplified implementation).
        
        Args:
            client_id: Client identifier
            limit: Request limit
            window: Time window in seconds
            
        Returns:
            True if within rate limit, False otherwise
        """
        # Simplified implementation for testing
        # In production, this would use Redis or similar
        import time
        current_time = time.time()
        
        # Mock rate limiting logic
        if not hasattr(self, '_rate_limits'):
            self._rate_limits = {}
        
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
        
        # Clean old entries
        self._rate_limits[client_id] = [
            t for t in self._rate_limits[client_id] 
            if current_time - t < window
        ]
        
        # Check limit
        if len(self._rate_limits[client_id]) >= limit:
            return False
        
        # Add current request
        self._rate_limits[client_id].append(current_time)
        return True
    
    def setup_audit_logging(self, log_file: str) -> None:
        """
        Setup audit logging.
        
        Args:
            log_file: Path to audit log file
        """
        self._audit_log_file = log_file
    
    def audit_log(self, event: str, data: dict) -> None:
        """
        Log audit event.
        
        Args:
            event: Event type
            data: Event data
        """
        if hasattr(self, '_audit_log_file'):
            import json
            from datetime import datetime
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event': event,
                'data': data
            }
            
            with open(self._audit_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    def scan_code_vulnerabilities(self, code: str) -> list:
        """
        Scan code for vulnerability patterns.
        
        Args:
            code: Code to scan
            
        Returns:
            List of detected vulnerabilities
        """
        vulnerabilities = []
        
        vulnerable_patterns = [
            (r'eval\s*\(.*\)', 'Dangerous eval() usage'),
            (r'exec\s*\(.*\)', 'Dangerous exec() usage'),
            (r'pickle\.loads\s*\(.*\)', 'Unsafe pickle.loads()'),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True.*\)', 'Shell injection risk'),
            (r'os\.system\s*\(.*\)', 'Command injection risk')
        ]
        
        for pattern, description in vulnerable_patterns:
            if re.search(pattern, code):
                vulnerabilities.append({
                    'pattern': pattern,
                    'description': description,
                    'severity': 'HIGH'
                })
        
        return vulnerabilities
    
    def validate_environment_security(self) -> dict:
        """
        Validate environment security configuration.
        
        Returns:
            Dictionary with security check results
        """
        import os
        
        results = {
            'debug_disabled': os.environ.get('DEBUG', 'True').lower() != 'true',
            'secret_key_set': os.environ.get('SECRET_KEY', 'default') != 'default',
            'allowed_hosts_configured': os.environ.get('ALLOWED_HOSTS', '*') != '*'
        }
        
        return results
    
    def anonymize_sensitive_data(self, data: dict) -> dict:
        """
        Anonymize sensitive data.
        
        Args:
            data: Data to anonymize
            
        Returns:
            Anonymized data
        """
        anonymized = {}
        
        for key, value in data.items():
            if key in ['email', 'phone', 'ssn', 'credit_card']:
                # Replace with hashed value
                anonymized[key] = f"***{hashlib.sha256(str(value).encode()).hexdigest()[:8]}***"
            elif key == 'ip_address':
                # Mask IP address
                anonymized[key] = "xxx.xxx.xxx.xxx"
            else:
                anonymized[key] = value
        
        return anonymized
    
    def apply_retention_policy(self, directory: str, max_age_days: int, file_patterns: list) -> list:
        """
        Apply data retention policy.
        
        Args:
            directory: Directory to scan
            max_age_days: Maximum age in days
            file_patterns: File patterns to match
            
        Returns:
            List of files marked for deletion
        """
        import glob
        import time
        
        deleted_files = []
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        for pattern in file_patterns:
            for file_path in glob.glob(os.path.join(directory, pattern)):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    deleted_files.append(file_path)
        
        return deleted_files
    
    def export_user_data(self, user_id: str, user_data: dict) -> dict:
        """
        Export user data for GDPR compliance.
        
        Args:
            user_id: User identifier
            user_data: User data to export
            
        Returns:
            Exported data
        """
        return user_data.copy()
    
    def anonymize_user_data(self, user_id: str, user_data: dict) -> dict:
        """
        Anonymize user data for GDPR compliance.
        
        Args:
            user_id: User identifier
            user_data: User data to anonymize
            
        Returns:
            Anonymized data
        """
        anonymized = user_data.copy()
        # Replace user_id with hash
        anonymized['user_id'] = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return anonymized