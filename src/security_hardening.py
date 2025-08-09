"""Security hardening module for IoT anomaly detection system."""

import hashlib
import hmac
import secrets
import time
import logging
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from enum import Enum
import threading
from collections import defaultdict, deque

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class SecurityLevel(Enum):
    """Security protection levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    DATA_INJECTION = "data_injection"
    MODEL_EVASION = "model_evasion"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    REPLAY_ATTACK = "replay_attack"
    MAN_IN_MIDDLE = "man_in_middle"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    timestamp: float
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    component: str = "unknown"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    mitigation_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_encryption: bool = True
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True
    enable_audit_logging: bool = True
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_window_minutes: int = 5
    
    # Authentication
    token_expiry_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    
    # Encryption
    encryption_key_rotation_hours: int = 24
    
    # Input validation
    max_input_size_mb: int = 10
    allowed_file_extensions: List[str] = field(default_factory=lambda: [".csv", ".json", ".txt"])
    
    # Monitoring
    alert_on_failed_auth: bool = True
    alert_on_rate_limit_exceeded: bool = True


class InputValidator:
    """Secure input validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dangerous patterns to block
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',                 # JavaScript injection
            r'on\w+\s*=',                  # Event handlers
            r'eval\s*\(',                  # Code evaluation
            r'exec\s*\(',                  # Code execution
            r'\bUNION\b.*\bSELECT\b',      # SQL injection
            r'\bDROP\b.*\bTABLE\b',        # SQL injection
            r'\.\./',                       # Path traversal
            r'\\x[0-9a-fA-F]{2}',          # Hex encoding
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Validate sensor data for security threats."""
        try:
            # Check data size
            data_size = len(json.dumps(data))
            if data_size > self.config.max_input_size_mb * 1024 * 1024:
                self.logger.warning(f"Input data too large: {data_size} bytes")
                return False
            
            # Validate structure
            if not isinstance(data, dict):
                return False
            
            # Check for required fields and types
            if "values" in data:
                values = data["values"]
                if not isinstance(values, list):
                    return False
                
                # Validate numeric values
                for value in values:
                    if not isinstance(value, (int, float)):
                        return False
                    
                    # Check for extreme values (potential attack)
                    if abs(value) > 1e10:
                        self.logger.warning(f"Extreme sensor value detected: {value}")
                        return False
            
            # Check string fields for injection attempts
            for key, value in data.items():
                if isinstance(value, str):
                    if not self._validate_string_content(value):
                        self.logger.warning(f"Malicious content detected in field {key}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return False
    
    def _validate_string_content(self, content: str) -> bool:
        """Validate string content for malicious patterns."""
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(content):
                return False
        
        # Check for excessive length
        if len(content) > 10000:  # 10KB limit for strings
            return False
        
        # Check for control characters (except standard whitespace)
        if any(ord(c) < 32 and c not in '\t\n\r' for c in content):
            return False
        
        return True
    
    def validate_file_upload(self, file_path: str, content: bytes) -> bool:
        """Validate uploaded files."""
        path = Path(file_path)
        
        # Check file extension
        if path.suffix.lower() not in self.config.allowed_file_extensions:
            self.logger.warning(f"Disallowed file extension: {path.suffix}")
            return False
        
        # Check file size
        if len(content) > self.config.max_input_size_mb * 1024 * 1024:
            self.logger.warning(f"File too large: {len(content)} bytes")
            return False
        
        # Check for executable headers
        dangerous_headers = [
            b'\x4D\x5A',  # PE executable
            b'\x7F\x45\x4C\x46',  # ELF executable
            b'\xFE\xED\xFA',  # Mach-O executable
            b'#!/',  # Script shebang
        ]
        
        for header in dangerous_headers:
            if content.startswith(header):
                self.logger.warning(f"Executable file detected: {file_path}")
                return False
        
        return True


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_counts: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, client_id: str, current_time: Optional[float] = None) -> bool:
        """Check if request is allowed under rate limits."""
        if current_time is None:
            current_time = time.time()
        
        with self.lock:
            # Clean old entries
            window_start = current_time - (self.config.rate_limit_window_minutes * 60)
            request_times = self.request_counts[client_id]
            
            while request_times and request_times[0] < window_start:
                request_times.popleft()
            
            # Check rate limit
            if len(request_times) >= self.config.rate_limit_requests_per_minute:
                self.logger.warning(f"Rate limit exceeded for client {client_id}")
                return False
            
            # Record this request
            request_times.append(current_time)
            return True
    
    def get_remaining_requests(self, client_id: str, current_time: Optional[float] = None) -> int:
        """Get number of remaining requests for client."""
        if current_time is None:
            current_time = time.time()
        
        with self.lock:
            window_start = current_time - (self.config.rate_limit_window_minutes * 60)
            request_times = self.request_counts[client_id]
            
            # Count recent requests
            recent_count = sum(1 for t in request_times if t >= window_start)
            return max(0, self.config.rate_limit_requests_per_minute - recent_count)


class AuthenticationManager:
    """Secure authentication and authorization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_tokens: Dict[str, Dict[str, Any]] = {}
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.locked_users: Dict[str, float] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def authenticate(self, username: str, password: str, client_ip: Optional[str] = None) -> Optional[str]:
        """Authenticate user and return token."""
        current_time = time.time()
        
        with self.lock:
            # Check if user is locked out
            if self._is_user_locked(username, current_time):
                self.logger.warning(f"Authentication attempted for locked user: {username}")
                return None
            
            # Simulate password check (replace with actual authentication)
            if self._verify_password(username, password):
                # Clear failed attempts on successful auth
                if username in self.failed_attempts:
                    del self.failed_attempts[username]
                
                # Generate secure token
                token = self._generate_token(username, client_ip)
                return token
            else:
                # Record failed attempt
                self._record_failed_attempt(username, current_time)
                return None
    
    def _is_user_locked(self, username: str, current_time: float) -> bool:
        """Check if user is currently locked out."""
        if username in self.locked_users:
            lockout_end = self.locked_users[username]
            if current_time < lockout_end:
                return True
            else:
                # Lockout expired
                del self.locked_users[username]
        return False
    
    def _record_failed_attempt(self, username: str, current_time: float) -> None:
        """Record failed authentication attempt."""
        attempts = self.failed_attempts[username]
        attempts.append(current_time)
        
        # Clean old attempts (older than lockout duration)
        cutoff_time = current_time - (self.config.lockout_duration_minutes * 60)
        self.failed_attempts[username] = [t for t in attempts if t > cutoff_time]
        
        # Check if lockout threshold exceeded
        if len(self.failed_attempts[username]) >= self.config.max_failed_attempts:
            lockout_end = current_time + (self.config.lockout_duration_minutes * 60)
            self.locked_users[username] = lockout_end
            self.logger.warning(f"User locked out due to failed attempts: {username}")
    
    def _verify_password(self, username: str, password: str) -> bool:
        """Verify password (placeholder - implement actual verification)."""
        # This should connect to your user database
        # For demo purposes, accept any non-empty password
        return len(password) > 0
    
    def _generate_token(self, username: str, client_ip: Optional[str] = None) -> str:
        """Generate secure authentication token."""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + (self.config.token_expiry_minutes * 60)
        
        self.active_tokens[token] = {
            "username": username,
            "client_ip": client_ip,
            "created_at": time.time(),
            "expires_at": expiry,
            "last_used": time.time()
        }
        
        return token
    
    def validate_token(self, token: str, client_ip: Optional[str] = None) -> Optional[str]:
        """Validate authentication token and return username."""
        current_time = time.time()
        
        with self.lock:
            if token not in self.active_tokens:
                return None
            
            token_data = self.active_tokens[token]
            
            # Check expiry
            if current_time > token_data["expires_at"]:
                del self.active_tokens[token]
                return None
            
            # Check IP if provided during token creation
            if (token_data.get("client_ip") and 
                client_ip and 
                token_data["client_ip"] != client_ip):
                self.logger.warning(f"Token used from different IP: {client_ip} vs {token_data['client_ip']}")
                return None
            
            # Update last used time
            token_data["last_used"] = current_time
            
            return token_data["username"]
    
    def revoke_token(self, token: str) -> bool:
        """Revoke authentication token."""
        with self.lock:
            if token in self.active_tokens:
                del self.active_tokens[token]
                return True
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens and return count removed."""
        current_time = time.time()
        expired_tokens = []
        
        with self.lock:
            for token, data in self.active_tokens.items():
                if current_time > data["expires_at"]:
                    expired_tokens.append(token)
            
            for token in expired_tokens:
                del self.active_tokens[token]
        
        return len(expired_tokens)


class EncryptionManager:
    """Data encryption and decryption."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.current_key: Optional[bytes] = None
        self.key_created_at: float = 0
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        if CRYPTOGRAPHY_AVAILABLE:
            self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption with new key."""
        with self.lock:
            self.current_key = Fernet.generate_key()
            self.key_created_at = time.time()
    
    def _should_rotate_key(self) -> bool:
        """Check if encryption key should be rotated."""
        if not self.current_key:
            return True
        
        age_hours = (time.time() - self.key_created_at) / 3600
        return age_hours > self.config.encryption_key_rotation_hours
    
    def encrypt_data(self, data: Union[str, bytes]) -> Optional[bytes]:
        """Encrypt sensitive data."""
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning("Cryptography library not available")
            return None
        
        with self.lock:
            if self._should_rotate_key():
                self._initialize_encryption()
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            try:
                fernet = Fernet(self.current_key)
                encrypted = fernet.encrypt(data)
                return encrypted
            except Exception as e:
                self.logger.error(f"Encryption failed: {e}")
                return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[bytes]:
        """Decrypt sensitive data."""
        if not CRYPTOGRAPHY_AVAILABLE:
            self.logger.warning("Cryptography library not available")
            return None
        
        with self.lock:
            if not self.current_key:
                self.logger.error("No encryption key available")
                return None
            
            try:
                fernet = Fernet(self.current_key)
                decrypted = fernet.decrypt(encrypted_data)
                return decrypted
            except Exception as e:
                self.logger.error(f"Decryption failed: {e}")
                return None
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        if CRYPTOGRAPHY_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            password_hash = kdf.derive(password.encode('utf-8'))
            return password_hash, salt
        else:
            # Fallback to hashlib
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            return password_hash, salt
    
    def verify_password(self, password: str, stored_hash: bytes, salt: bytes) -> bool:
        """Verify password against stored hash."""
        calculated_hash, _ = self.hash_password(password, salt)
        return hmac.compare_digest(calculated_hash, stored_hash)


class SecurityMonitor:
    """Security monitoring and threat detection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security_events: List[SecurityEvent] = []
        self.event_handlers: List[Callable[[SecurityEvent], None]] = []
        self.threat_counters: Dict[ThreatType, int] = defaultdict(int)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def add_event_handler(self, handler: Callable[[SecurityEvent], None]) -> None:
        """Add security event handler."""
        self.event_handlers.append(handler)
    
    def report_security_event(
        self,
        threat_type: ThreatType,
        severity: SecurityLevel,
        component: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        blocked: bool = False
    ) -> SecurityEvent:
        """Report security event."""
        event = SecurityEvent(
            event_id=f"sec_{int(time.time() * 1000)}_{secrets.token_hex(4)}",
            timestamp=time.time(),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            component=component,
            description=description,
            metadata=metadata or {},
            blocked=blocked
        )
        
        with self.lock:
            self.security_events.append(event)
            self.threat_counters[threat_type] += 1
            
            # Maintain event history size
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-5000:]
        
        # Log event
        self._log_security_event(event)
        
        # Call event handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Security event handler failed: {e}")
        
        return event
    
    def _log_security_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }.get(event.severity, logging.WARNING)
        
        message = (
            f"SECURITY [{event.threat_type.value}] "
            f"{event.component}: {event.description}"
        )
        
        if event.source_ip:
            message += f" (IP: {event.source_ip})"
        
        if event.blocked:
            message += " [BLOCKED]"
        
        self.logger.log(log_level, message)
    
    def get_threat_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get threat summary for time window."""
        current_time = time.time()
        window_start = current_time - (time_window_hours * 3600)
        
        with self.lock:
            recent_events = [e for e in self.security_events if e.timestamp >= window_start]
        
        # Count by threat type
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        blocked_count = 0
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.severity.value] += 1
            if event.blocked:
                blocked_count += 1
        
        return {
            "time_window_hours": time_window_hours,
            "total_events": len(recent_events),
            "blocked_events": blocked_count,
            "threat_counts": dict(threat_counts),
            "severity_counts": dict(severity_counts),
            "most_common_threat": max(threat_counts.items(), key=lambda x: x[1])[0] if threat_counts else None
        }
    
    def detect_anomalous_patterns(self) -> List[str]:
        """Detect anomalous security patterns."""
        alerts = []
        
        # Check for burst of events from same IP
        ip_counts = defaultdict(int)
        recent_time = time.time() - 300  # Last 5 minutes
        
        with self.lock:
            for event in self.security_events:
                if event.timestamp >= recent_time and event.source_ip:
                    ip_counts[event.source_ip] += 1
        
        for ip, count in ip_counts.items():
            if count > 10:  # More than 10 events in 5 minutes
                alerts.append(f"High event frequency from IP {ip}: {count} events")
        
        # Check for escalating severity
        recent_events = [e for e in self.security_events if e.timestamp >= recent_time]
        if len(recent_events) >= 5:
            severity_trend = [e.severity for e in recent_events[-5:]]
            if all(s.value in ["high", "critical"] for s in severity_trend):
                alerts.append("Escalating security threat severity detected")
        
        return alerts


class SecurityHardening:
    """Main security hardening system."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.input_validator = InputValidator(self.config) if self.config.enable_input_validation else None
        self.rate_limiter = RateLimiter(self.config) if self.config.enable_rate_limiting else None
        self.auth_manager = AuthenticationManager(self.config) if self.config.enable_authentication else None
        self.encryption_manager = EncryptionManager(self.config) if self.config.enable_encryption else None
        self.security_monitor = SecurityMonitor(self.config)
        
        # Periodic cleanup
        self._setup_periodic_cleanup()
    
    def _setup_periodic_cleanup(self) -> None:
        """Setup periodic cleanup tasks."""
        def cleanup_task():
            try:
                if self.auth_manager:
                    expired = self.auth_manager.cleanup_expired_tokens()
                    if expired > 0:
                        self.logger.info(f"Cleaned up {expired} expired tokens")
            except Exception as e:
                self.logger.error(f"Cleanup task failed: {e}")
        
        # Schedule periodic cleanup (simplified - use proper scheduler in production)
        threading.Timer(3600, cleanup_task).start()  # Run every hour
    
    def validate_and_process_data(
        self,
        data: Dict[str, Any],
        client_id: str,
        client_ip: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive security validation and processing."""
        try:
            # Rate limiting
            if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
                self.security_monitor.report_security_event(
                    ThreatType.DENIAL_OF_SERVICE,
                    SecurityLevel.MEDIUM,
                    "rate_limiter",
                    f"Rate limit exceeded for client {client_id}",
                    source_ip=client_ip,
                    blocked=True
                )
                raise SecurityError("Rate limit exceeded")
            
            # Authentication
            username = None
            if self.auth_manager and auth_token:
                username = self.auth_manager.validate_token(auth_token, client_ip)
                if not username:
                    self.security_monitor.report_security_event(
                        ThreatType.UNAUTHORIZED_ACCESS,
                        SecurityLevel.HIGH,
                        "auth_manager",
                        "Invalid or expired authentication token",
                        source_ip=client_ip,
                        blocked=True
                    )
                    raise SecurityError("Authentication failed")
            
            # Input validation
            if self.input_validator and not self.input_validator.validate_sensor_data(data):
                self.security_monitor.report_security_event(
                    ThreatType.DATA_INJECTION,
                    SecurityLevel.HIGH,
                    "input_validator",
                    "Malicious input detected",
                    source_ip=client_ip,
                    user_id=username,
                    blocked=True
                )
                raise SecurityError("Input validation failed")
            
            # Encrypt sensitive data if needed
            processed_data = data.copy()
            if self.encryption_manager and "sensitive_values" in data:
                encrypted_values = self.encryption_manager.encrypt_data(
                    json.dumps(data["sensitive_values"])
                )
                if encrypted_values:
                    processed_data["sensitive_values"] = encrypted_values.hex()
                    processed_data["_encrypted"] = True
            
            # Log successful processing
            self.security_monitor.report_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,  # Placeholder for successful access
                SecurityLevel.LOW,
                "security_hardening",
                "Data processed successfully",
                source_ip=client_ip,
                user_id=username
            )
            
            return processed_data
            
        except SecurityError:
            raise
        except Exception as e:
            self.security_monitor.report_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                "security_hardening",
                f"Security processing error: {str(e)}",
                source_ip=client_ip,
                user_id=username
            )
            raise SecurityError(f"Security processing failed: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "configuration": {
                "encryption_enabled": self.config.enable_encryption,
                "authentication_enabled": self.config.enable_authentication,
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "input_validation_enabled": self.config.enable_input_validation
            },
            "threat_summary": self.security_monitor.get_threat_summary(),
            "anomaly_alerts": self.security_monitor.detect_anomalous_patterns(),
            "active_tokens": len(self.auth_manager.active_tokens) if self.auth_manager else 0
        }


class SecurityError(Exception):
    """Security-related exception."""
    pass


# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Security Hardening System")
    parser.add_argument("--config", help="Security configuration file")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    # Initialize security system
    config = SecurityConfig()
    security = SecurityHardening(config)
    
    if args.test_mode:
        print("Running security hardening tests...")
        
        # Test data validation
        test_data = {
            "values": [1.0, 2.0, 3.0],
            "timestamp": time.time(),
            "sensor_id": "test_sensor"
        }
        
        try:
            processed = security.validate_and_process_data(
                test_data,
                client_id="test_client",
                client_ip="127.0.0.1"
            )
            print("✓ Valid data processed successfully")
        except SecurityError as e:
            print(f"✗ Security error: {e}")
        
        # Test malicious data
        malicious_data = {
            "values": [1.0, 2.0, 3.0],
            "script": "<script>alert('xss')</script>",
            "timestamp": time.time()
        }
        
        try:
            security.validate_and_process_data(
                malicious_data,
                client_id="test_client",
                client_ip="127.0.0.1"
            )
            print("✗ Malicious data should have been blocked")
        except SecurityError:
            print("✓ Malicious data correctly blocked")
        
        # Print security status
        status = security.get_security_status()
        print(f"Security Status: {json.dumps(status, indent=2)}")
        
    else:
        print("Security hardening system initialized")
        print("Use --test-mode to run security tests")