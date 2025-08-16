"""Advanced Security Framework for IoT Anomaly Detection System.

This module provides comprehensive security measures including authentication,
authorization, encryption, audit logging, and threat detection.
"""

import base64
import json
import logging
import secrets
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .logging_config import get_logger


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityCredential:
    """Security credential definition."""
    id: str
    user_id: str
    credential_type: AuthenticationMethod
    credential_data: str  # Hashed/encrypted
    permissions: List[str]
    security_level: SecurityLevel
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0
    expires_at: float = 0.0
    is_active: bool = True


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    event_type: str
    user_id: str
    resource: str
    action: str
    result: str
    threat_level: ThreatLevel
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source_ip: str = ""
    user_agent: str = ""


@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: str
    permissions: List[str]
    security_level: SecurityLevel
    authentication_method: AuthenticationMethod
    session_token: str
    expires_at: float


class AdvancedSecurityFramework:
    """Comprehensive security framework for IoT anomaly detection."""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry_hours: int = 24,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        enable_audit_logging: bool = True
    ):
        """Initialize the security framework.
        
        Args:
            secret_key: Secret key for JWT and encryption
            token_expiry_hours: JWT token expiry time in hours
            max_failed_attempts: Maximum failed authentication attempts
            lockout_duration_minutes: Account lockout duration after max failures
            enable_audit_logging: Enable comprehensive audit logging
        """
        self.logger = get_logger(__name__)
        self.secret_key = secret_key or self._generate_secret_key()
        self.token_expiry_hours = token_expiry_hours
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.enable_audit_logging = enable_audit_logging

        # Security state
        self.credentials: Dict[str, SecurityCredential] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.locked_accounts: Dict[str, float] = {}

        # Encryption
        self.cipher_suite = self._initialize_encryption()

        # Audit logging
        self.security_events = deque(maxlen=10000)
        self.threat_detection_rules: List[Callable] = []

        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Threading
        self.security_lock = threading.RLock()

        # Initialize default security rules
        self._initialize_threat_detection()

        self.logger.info("Advanced security framework initialized")

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()

    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption cipher."""
        key = base64.urlsafe_b64encode(
            PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'salt_',  # In production, use random salt
                iterations=100000,
            ).derive(self.secret_key.encode())
        )
        return Fernet(key)

    def _initialize_threat_detection(self) -> None:
        """Initialize threat detection rules."""
        self.threat_detection_rules = [
            self._detect_brute_force_attack,
            self._detect_unusual_access_pattern,
            self._detect_privilege_escalation,
            self._detect_suspicious_api_usage
        ]

    def create_user_credential(
        self,
        user_id: str,
        password: str,
        permissions: List[str],
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        auth_method: AuthenticationMethod = AuthenticationMethod.JWT_TOKEN
    ) -> str:
        """Create a new user credential.
        
        Args:
            user_id: Unique user identifier
            password: User password (will be hashed)
            permissions: List of permissions for the user
            security_level: Security clearance level
            auth_method: Authentication method
            
        Returns:
            Credential ID
        """
        with self.security_lock:
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            # Create credential
            credential_id = self._generate_credential_id()
            credential = SecurityCredential(
                id=credential_id,
                user_id=user_id,
                credential_type=auth_method,
                credential_data=password_hash.decode('utf-8'),
                permissions=permissions,
                security_level=security_level,
                expires_at=time.time() + (365 * 24 * 3600)  # 1 year expiry
            )

            self.credentials[credential_id] = credential

            # Audit log
            self._log_security_event(
                event_type="credential_created",
                user_id=user_id,
                resource="user_credential",
                action="create",
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"credential_id": credential_id, "auth_method": auth_method.value}
            )

            self.logger.info(f"Created credential for user: {user_id}")
            return credential_id

    def authenticate_user(
        self,
        user_id: str,
        password: str,
        source_ip: str = "",
        user_agent: str = ""
    ) -> Optional[str]:
        """Authenticate a user and return a session token.
        
        Args:
            user_id: User identifier
            password: User password
            source_ip: Source IP address
            user_agent: User agent string
            
        Returns:
            JWT session token if authentication successful, None otherwise
        """
        with self.security_lock:
            # Check if account is locked
            if self._is_account_locked(user_id):
                self._log_security_event(
                    event_type="authentication_blocked",
                    user_id=user_id,
                    resource="authentication",
                    action="login",
                    result="blocked_locked_account",
                    threat_level=ThreatLevel.HIGH,
                    details={"reason": "account_locked"},
                    source_ip=source_ip,
                    user_agent=user_agent
                )
                return None

            # Find credential
            credential = self._find_credential_by_user(user_id)
            if not credential or not credential.is_active:
                self._record_failed_attempt(user_id)
                self._log_security_event(
                    event_type="authentication_failed",
                    user_id=user_id,
                    resource="authentication",
                    action="login",
                    result="invalid_user",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "user_not_found"},
                    source_ip=source_ip,
                    user_agent=user_agent
                )
                return None

            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), credential.credential_data.encode('utf-8')):
                self._record_failed_attempt(user_id)
                self._log_security_event(
                    event_type="authentication_failed",
                    user_id=user_id,
                    resource="authentication",
                    action="login",
                    result="invalid_password",
                    threat_level=ThreatLevel.MEDIUM,
                    details={"reason": "invalid_password"},
                    source_ip=source_ip,
                    user_agent=user_agent
                )
                return None

            # Check credential expiry
            if credential.expires_at > 0 and time.time() > credential.expires_at:
                self._log_security_event(
                    event_type="authentication_failed",
                    user_id=user_id,
                    resource="authentication",
                    action="login",
                    result="credential_expired",
                    threat_level=ThreatLevel.LOW,
                    details={"reason": "credential_expired"},
                    source_ip=source_ip,
                    user_agent=user_agent
                )
                return None

            # Generate session token
            session_token = self._generate_jwt_token(credential)

            # Create security context
            security_context = SecurityContext(
                user_id=user_id,
                permissions=credential.permissions,
                security_level=credential.security_level,
                authentication_method=credential.credential_type,
                session_token=session_token,
                expires_at=time.time() + (self.token_expiry_hours * 3600)
            )

            self.active_sessions[session_token] = security_context

            # Update credential last used
            credential.last_used = time.time()

            # Clear failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]

            # Audit log
            self._log_security_event(
                event_type="authentication_success",
                user_id=user_id,
                resource="authentication",
                action="login",
                result="success",
                threat_level=ThreatLevel.LOW,
                details={"session_token": session_token[:10] + "..."},
                source_ip=source_ip,
                user_agent=user_agent
            )

            # Run threat detection
            self._run_threat_detection(user_id, source_ip, user_agent)

            self.logger.info(f"User authenticated successfully: {user_id}")
            return session_token

    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate a session token and return security context.
        
        Args:
            session_token: JWT session token
            
        Returns:
            Security context if valid, None otherwise
        """
        try:
            # Verify JWT token
            payload = jwt.decode(session_token, self.secret_key, algorithms=['HS256'])

            # Check if session exists and is not expired
            if session_token in self.active_sessions:
                context = self.active_sessions[session_token]
                if time.time() < context.expires_at:
                    return context
                else:
                    # Session expired
                    del self.active_sessions[session_token]

            return None

        except jwt.InvalidTokenError:
            return None

    def authorize_action(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str,
        required_security_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> bool:
        """Authorize an action based on security context.
        
        Args:
            security_context: Current security context
            resource: Resource being accessed
            action: Action being performed
            required_security_level: Minimum required security level
            
        Returns:
            True if authorized, False otherwise
        """
        # Check security level
        if not self._has_sufficient_security_level(security_context.security_level, required_security_level):
            self._log_security_event(
                event_type="authorization_failed",
                user_id=security_context.user_id,
                resource=resource,
                action=action,
                result="insufficient_security_level",
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "user_level": security_context.security_level.value,
                    "required_level": required_security_level.value
                }
            )
            return False

        # Check permissions
        required_permission = f"{resource}:{action}"
        if required_permission not in security_context.permissions and "admin:*" not in security_context.permissions:
            self._log_security_event(
                event_type="authorization_failed",
                user_id=security_context.user_id,
                resource=resource,
                action=action,
                result="insufficient_permissions",
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "required_permission": required_permission,
                    "user_permissions": security_context.permissions
                }
            )
            return False

        # Log successful authorization
        self._log_security_event(
            event_type="authorization_success",
            user_id=security_context.user_id,
            resource=resource,
            action=action,
            result="success",
            threat_level=ThreatLevel.LOW,
            details={"permission": required_permission}
        )

        return True

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data as base64 string
        """
        encrypted_bytes = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted data
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        return self.cipher_suite.decrypt(encrypted_bytes).decode()

    def check_rate_limit(self, user_id: str, requests_per_minute: int = 60) -> bool:
        """Check if user has exceeded rate limit.
        
        Args:
            user_id: User identifier
            requests_per_minute: Maximum requests per minute
            
        Returns:
            True if within rate limit, False otherwise
        """
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago

        # Clean old entries
        user_requests = self.rate_limits[user_id]
        while user_requests and user_requests[0] < cutoff_time:
            user_requests.popleft()

        # Check limit
        if len(user_requests) >= requests_per_minute:
            self._log_security_event(
                event_type="rate_limit_exceeded",
                user_id=user_id,
                resource="api",
                action="request",
                result="rate_limit_exceeded",
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "requests_count": len(user_requests),
                    "limit": requests_per_minute
                }
            )
            return False

        # Add current request
        user_requests.append(current_time)
        return True

    def _generate_credential_id(self) -> str:
        """Generate a unique credential ID."""
        return f"cred_{secrets.token_urlsafe(16)}"

    def _generate_jwt_token(self, credential: SecurityCredential) -> str:
        """Generate a JWT token for the credential."""
        payload = {
            'user_id': credential.user_id,
            'credential_id': credential.id,
            'permissions': credential.permissions,
            'security_level': credential.security_level.value,
            'iat': time.time(),
            'exp': time.time() + (self.token_expiry_hours * 3600)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def _find_credential_by_user(self, user_id: str) -> Optional[SecurityCredential]:
        """Find credential by user ID."""
        for credential in self.credentials.values():
            if credential.user_id == user_id:
                return credential
        return None

    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if user_id in self.locked_accounts:
            lock_time = self.locked_accounts[user_id]
            if time.time() - lock_time < (self.lockout_duration_minutes * 60):
                return True
            else:
                # Lock expired
                del self.locked_accounts[user_id]
        return False

    def _record_failed_attempt(self, user_id: str) -> None:
        """Record a failed authentication attempt."""
        current_time = time.time()
        self.failed_attempts[user_id].append(current_time)

        # Keep only recent attempts (last hour)
        cutoff_time = current_time - 3600
        self.failed_attempts[user_id] = [
            t for t in self.failed_attempts[user_id] if t > cutoff_time
        ]

        # Check if account should be locked
        if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.locked_accounts[user_id] = current_time
            self._log_security_event(
                event_type="account_locked",
                user_id=user_id,
                resource="authentication",
                action="lock",
                result="max_failed_attempts",
                threat_level=ThreatLevel.HIGH,
                details={
                    "failed_attempts": len(self.failed_attempts[user_id]),
                    "max_attempts": self.max_failed_attempts
                }
            )

    def _has_sufficient_security_level(self, user_level: SecurityLevel, required_level: SecurityLevel) -> bool:
        """Check if user has sufficient security level."""
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.SECRET: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        return level_hierarchy[user_level] >= level_hierarchy[required_level]

    def _log_security_event(
        self,
        event_type: str,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        threat_level: ThreatLevel,
        details: Dict[str, Any],
        source_ip: str = "",
        user_agent: str = ""
    ) -> None:
        """Log a security event."""
        if not self.enable_audit_logging:
            return

        event = SecurityEvent(
            event_id=f"sec_{secrets.token_urlsafe(8)}",
            event_type=event_type,
            user_id=user_id,
            resource=resource,
            action=action,
            result=result,
            threat_level=threat_level,
            details=details,
            source_ip=source_ip,
            user_agent=user_agent
        )

        self.security_events.append(event)

        # Log to standard logger
        self.logger.info(
            f"Security Event: {event_type} - {user_id} - {resource}:{action} - {result} - {threat_level.value}"
        )

    def _run_threat_detection(self, user_id: str, source_ip: str, user_agent: str) -> None:
        """Run threat detection rules."""
        for rule in self.threat_detection_rules:
            try:
                rule(user_id, source_ip, user_agent)
            except Exception as e:
                self.logger.error(f"Threat detection rule failed: {e}")

    def _detect_brute_force_attack(self, user_id: str, source_ip: str, user_agent: str) -> None:
        """Detect potential brute force attacks."""
        # Check for multiple failed attempts from same IP
        recent_events = [
            event for event in self.security_events
            if (event.source_ip == source_ip and
                event.event_type == "authentication_failed" and
                time.time() - event.timestamp < 600)  # Last 10 minutes
        ]

        if len(recent_events) >= 10:
            self._log_security_event(
                event_type="threat_detected",
                user_id=user_id,
                resource="security",
                action="brute_force_detection",
                result="potential_brute_force",
                threat_level=ThreatLevel.HIGH,
                details={
                    "source_ip": source_ip,
                    "failed_attempts": len(recent_events),
                    "time_window": "10_minutes"
                },
                source_ip=source_ip,
                user_agent=user_agent
            )

    def _detect_unusual_access_pattern(self, user_id: str, source_ip: str, user_agent: str) -> None:
        """Detect unusual access patterns."""
        # Check for access from new IP for this user
        user_events = [
            event for event in self.security_events
            if (event.user_id == user_id and
                event.event_type == "authentication_success" and
                time.time() - event.timestamp < (7 * 24 * 3600))  # Last 7 days
        ]

        known_ips = set(event.source_ip for event in user_events)
        if source_ip not in known_ips and len(user_events) > 0:
            self._log_security_event(
                event_type="threat_detected",
                user_id=user_id,
                resource="security",
                action="unusual_access_detection",
                result="new_ip_address",
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "new_ip": source_ip,
                    "known_ips": list(known_ips)
                },
                source_ip=source_ip,
                user_agent=user_agent
            )

    def _detect_privilege_escalation(self, user_id: str, source_ip: str, user_agent: str) -> None:
        """Detect potential privilege escalation attempts."""
        # Check for authorization failures indicating privilege escalation attempts
        recent_auth_failures = [
            event for event in self.security_events
            if (event.user_id == user_id and
                event.event_type == "authorization_failed" and
                time.time() - event.timestamp < 300)  # Last 5 minutes
        ]

        if len(recent_auth_failures) >= 5:
            self._log_security_event(
                event_type="threat_detected",
                user_id=user_id,
                resource="security",
                action="privilege_escalation_detection",
                result="multiple_authorization_failures",
                threat_level=ThreatLevel.HIGH,
                details={
                    "failed_authorizations": len(recent_auth_failures),
                    "time_window": "5_minutes"
                },
                source_ip=source_ip,
                user_agent=user_agent
            )

    def _detect_suspicious_api_usage(self, user_id: str, source_ip: str, user_agent: str) -> None:
        """Detect suspicious API usage patterns."""
        # Check for unusually high API usage
        recent_requests = [
            event for event in self.security_events
            if (event.user_id == user_id and
                time.time() - event.timestamp < 300)  # Last 5 minutes
        ]

        if len(recent_requests) >= 100:  # 100 requests in 5 minutes
            self._log_security_event(
                event_type="threat_detected",
                user_id=user_id,
                resource="security",
                action="suspicious_api_usage",
                result="high_request_rate",
                threat_level=ThreatLevel.MEDIUM,
                details={
                    "request_count": len(recent_requests),
                    "time_window": "5_minutes"
                },
                source_ip=source_ip,
                user_agent=user_agent
            )

    def get_security_audit_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security audit report.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Comprehensive security audit report
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]

        # Aggregate statistics
        event_counts = defaultdict(int)
        threat_counts = defaultdict(int)
        user_activity = defaultdict(int)

        for event in recent_events:
            event_counts[event.event_type] += 1
            threat_counts[event.threat_level.value] += 1
            user_activity[event.user_id] += 1

        report = {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'event_breakdown': dict(event_counts),
            'threat_level_breakdown': dict(threat_counts),
            'top_active_users': dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
            'active_sessions': len(self.active_sessions),
            'locked_accounts': len(self.locked_accounts),
            'failed_attempt_users': len(self.failed_attempts),
            'security_summary': {
                'authentication_success_rate': 0.0,
                'authorization_success_rate': 0.0,
                'threat_detection_rate': 0.0
            }
        }

        # Calculate success rates
        auth_success = event_counts.get('authentication_success', 0)
        auth_failed = event_counts.get('authentication_failed', 0)
        if auth_success + auth_failed > 0:
            report['security_summary']['authentication_success_rate'] = auth_success / (auth_success + auth_failed)

        authz_success = event_counts.get('authorization_success', 0)
        authz_failed = event_counts.get('authorization_failed', 0)
        if authz_success + authz_failed > 0:
            report['security_summary']['authorization_success_rate'] = authz_success / (authz_success + authz_failed)

        threats_detected = event_counts.get('threat_detected', 0)
        if len(recent_events) > 0:
            report['security_summary']['threat_detection_rate'] = threats_detected / len(recent_events)

        return report

    def export_audit_log(self, filepath: str, hours: int = 24) -> None:
        """Export security audit log to file.
        
        Args:
            filepath: Output file path
            hours: Number of hours to include
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]

        audit_data = {
            'exported_at': datetime.now().isoformat(),
            'period_hours': hours,
            'total_events': len(recent_events),
            'events': [
                {
                    'event_id': event.event_id,
                    'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'resource': event.resource,
                    'action': event.action,
                    'result': event.result,
                    'threat_level': event.threat_level.value,
                    'source_ip': event.source_ip,
                    'user_agent': event.user_agent,
                    'details': event.details
                }
                for event in recent_events
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)

        self.logger.info(f"Security audit log exported to {filepath}")


# CLI Interface
def main() -> None:
    """CLI entry point for security framework."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Security Framework"
    )
    parser.add_argument(
        "--create-user",
        nargs=3,
        metavar=('USER_ID', 'PASSWORD', 'PERMISSIONS'),
        help="Create a new user with permissions (comma-separated)"
    )
    parser.add_argument(
        "--authenticate",
        nargs=2,
        metavar=('USER_ID', 'PASSWORD'),
        help="Authenticate user and get session token"
    )
    parser.add_argument(
        "--audit-report",
        type=int,
        default=24,
        help="Generate audit report for last N hours"
    )
    parser.add_argument(
        "--export-audit",
        help="Export audit log to file"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)

    # Create security framework
    security = AdvancedSecurityFramework()

    if args.create_user:
        user_id, password, permissions_str = args.create_user
        permissions = permissions_str.split(',')

        credential_id = security.create_user_credential(
            user_id=user_id,
            password=password,
            permissions=permissions,
            security_level=SecurityLevel.INTERNAL
        )

        print(f"User created successfully. Credential ID: {credential_id}")

    elif args.authenticate:
        user_id, password = args.authenticate

        token = security.authenticate_user(user_id, password)
        if token:
            print(f"Authentication successful. Token: {token}")
        else:
            print("Authentication failed.")

    elif args.audit_report:
        report = security.get_security_audit_report(args.audit_report)
        print("\n" + "="*50)
        print("SECURITY AUDIT REPORT")
        print("="*50)
        print(json.dumps(report, indent=2))

    if args.export_audit:
        security.export_audit_log(args.export_audit, 24)
        print(f"Audit log exported to: {args.export_audit}")


if __name__ == "__main__":
    main()
