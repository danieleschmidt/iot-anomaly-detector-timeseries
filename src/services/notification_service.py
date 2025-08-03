"""
Notification Service

Business logic for alert generation and notification delivery.
"""

import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import requests
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Supported notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationService:
    """
    Service for managing notifications and alerts.
    
    Handles alert generation, routing, and delivery through
    various notification channels.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_rate_limiting: bool = True,
        enable_batching: bool = True
    ):
        """
        Initialize the notification service.
        
        Args:
            config_path: Path to notification configuration file
            enable_rate_limiting: Whether to enable rate limiting
            enable_batching: Whether to batch notifications
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_batching = enable_batching
        
        # Alert history and rate limiting
        self._alert_history = deque(maxlen=1000)
        self._rate_limits = {
            NotificationChannel.EMAIL: {'max_per_hour': 100, 'sent': []},
            NotificationChannel.SLACK: {'max_per_hour': 200, 'sent': []},
            NotificationChannel.WEBHOOK: {'max_per_hour': 500, 'sent': []},
            NotificationChannel.SMS: {'max_per_hour': 50, 'sent': []},
            NotificationChannel.TEAMS: {'max_per_hour': 200, 'sent': []}
        }
        
        # Batching queues
        self._batch_queues = {channel: [] for channel in NotificationChannel}
        self._batch_settings = {
            'max_batch_size': 10,
            'batch_interval_seconds': 60
        }
        
        # Notification templates
        self._templates = self._load_templates()
        
        # Statistics
        self._stats = {
            'total_alerts': 0,
            'sent_notifications': 0,
            'failed_notifications': 0,
            'suppressed_notifications': 0
        }
    
    def send_anomaly_alert(
        self,
        anomaly_data: Dict[str, Any],
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[str, Any]:
        """
        Send anomaly detection alert.
        
        Args:
            anomaly_data: Anomaly detection results
            severity: Alert severity level
            channels: Notification channels to use
            
        Returns:
            Notification status and results
        """
        # Prepare alert
        alert = self._prepare_anomaly_alert(anomaly_data, severity)
        
        # Default channels if not specified
        if not channels:
            channels = self._get_default_channels(severity)
        
        # Check for duplicate alerts
        if self._is_duplicate_alert(alert):
            logger.info("Suppressing duplicate alert")
            self._stats['suppressed_notifications'] += 1
            return {
                'status': 'suppressed',
                'reason': 'duplicate_alert',
                'alert_id': alert['id']
            }
        
        # Send notifications
        results = self._send_to_channels(alert, channels)
        
        # Record alert
        self._record_alert(alert, results)
        
        return {
            'status': 'sent' if any(r['success'] for r in results.values()) else 'failed',
            'alert_id': alert['id'],
            'severity': severity.value,
            'channels_attempted': [c.value for c in channels],
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def send_system_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: AlertSeverity = AlertSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """
        Send system-level alert.
        
        Args:
            alert_type: Type of system alert
            message: Alert message
            details: Additional alert details
            severity: Alert severity
            
        Returns:
            Notification status
        """
        alert = {
            'id': self._generate_alert_id(),
            'type': 'system',
            'alert_type': alert_type,
            'message': message,
            'details': details or {},
            'severity': severity.value,
            'timestamp': datetime.now().isoformat()
        }
        
        channels = self._get_default_channels(severity)
        results = self._send_to_channels(alert, channels)
        
        self._record_alert(alert, results)
        
        return {
            'status': 'sent' if any(r['success'] for r in results.values()) else 'failed',
            'alert_id': alert['id'],
            'results': results
        }
    
    def batch_send(
        self,
        alerts: List[Dict[str, Any]],
        channel: NotificationChannel
    ) -> Dict[str, Any]:
        """
        Send batched notifications.
        
        Args:
            alerts: List of alerts to send
            channel: Notification channel
            
        Returns:
            Batch send results
        """
        logger.info(f"Sending batch of {len(alerts)} alerts to {channel.value}")
        
        if self.enable_batching:
            # Add to batch queue
            self._batch_queues[channel].extend(alerts)
            
            # Check if batch is ready
            if len(self._batch_queues[channel]) >= self._batch_settings['max_batch_size']:
                return self._flush_batch(channel)
            else:
                return {
                    'status': 'queued',
                    'queued_alerts': len(self._batch_queues[channel]),
                    'channel': channel.value
                }
        else:
            # Send immediately
            results = []
            for alert in alerts:
                result = self._send_to_channel(alert, channel)
                results.append(result)
            
            return {
                'status': 'sent',
                'total_alerts': len(alerts),
                'successful': sum(1 for r in results if r['success']),
                'failed': sum(1 for r in results if not r['success']),
                'results': results
            }
    
    def configure_channel(
        self,
        channel: NotificationChannel,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Configure a notification channel.
        
        Args:
            channel: Channel to configure
            config: Channel configuration
            
        Returns:
            Configuration status
        """
        logger.info(f"Configuring {channel.value} channel")
        
        # Validate configuration
        if not self._validate_channel_config(channel, config):
            return {
                'status': 'failed',
                'error': 'Invalid configuration',
                'channel': channel.value
            }
        
        # Store configuration
        if 'channels' not in self.config:
            self.config['channels'] = {}
        
        self.config['channels'][channel.value] = config
        
        # Test connection if possible
        test_result = self._test_channel(channel, config)
        
        return {
            'status': 'configured',
            'channel': channel.value,
            'test_result': test_result,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            severity: Filter by severity
            
        Returns:
            List of historical alerts
        """
        alerts = list(self._alert_history)
        
        # Apply filters
        if start_time:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a['timestamp']) >= start_time
            ]
        
        if end_time:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a['timestamp']) <= end_time
            ]
        
        if severity:
            alerts = [
                a for a in alerts
                if a.get('severity') == severity.value
            ]
        
        return alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notification service statistics.
        
        Returns:
            Service statistics
        """
        # Calculate channel statistics
        channel_stats = {}
        for channel in NotificationChannel:
            sent_count = len(self._rate_limits[channel]['sent'])
            channel_stats[channel.value] = {
                'sent_last_hour': sent_count,
                'rate_limit': self._rate_limits[channel]['max_per_hour'],
                'queued': len(self._batch_queues[channel])
            }
        
        return {
            'overall': self._stats,
            'channels': channel_stats,
            'alert_history_size': len(self._alert_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def create_alert_rule(
        self,
        rule_name: str,
        condition: Dict[str, Any],
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create custom alert rule.
        
        Args:
            rule_name: Name of the rule
            condition: Rule condition definition
            actions: Actions to take when rule triggers
            
        Returns:
            Rule creation status
        """
        rule = {
            'name': rule_name,
            'condition': condition,
            'actions': actions,
            'created_at': datetime.now().isoformat(),
            'enabled': True,
            'triggered_count': 0
        }
        
        # Store rule (in production, would persist to database)
        if 'rules' not in self.config:
            self.config['rules'] = {}
        
        self.config['rules'][rule_name] = rule
        
        logger.info(f"Created alert rule: {rule_name}")
        
        return {
            'status': 'created',
            'rule_name': rule_name,
            'rule': rule
        }
    
    def evaluate_alert_rules(
        self,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate alert rules against data.
        
        Args:
            data: Data to evaluate
            
        Returns:
            Triggered rules and actions
        """
        triggered_rules = []
        
        if 'rules' not in self.config:
            return triggered_rules
        
        for rule_name, rule in self.config['rules'].items():
            if not rule.get('enabled', True):
                continue
            
            # Evaluate condition (simplified)
            if self._evaluate_condition(rule['condition'], data):
                logger.info(f"Rule triggered: {rule_name}")
                
                # Execute actions
                for action in rule['actions']:
                    self._execute_action(action, data)
                
                rule['triggered_count'] += 1
                triggered_rules.append({
                    'rule_name': rule_name,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                })
        
        return triggered_rules
    
    def _prepare_anomaly_alert(
        self,
        anomaly_data: Dict[str, Any],
        severity: AlertSeverity
    ) -> Dict[str, Any]:
        """Prepare anomaly alert from detection results."""
        return {
            'id': self._generate_alert_id(),
            'type': 'anomaly',
            'severity': severity.value,
            'timestamp': datetime.now().isoformat(),
            'anomaly_count': anomaly_data.get('anomalies_detected', 0),
            'threshold': anomaly_data.get('threshold'),
            'statistics': anomaly_data.get('statistics', {}),
            'model_version': anomaly_data.get('model_version'),
            'processing_time': anomaly_data.get('processing_time'),
            'message': self._generate_alert_message(anomaly_data, severity)
        }
    
    def _generate_alert_message(
        self,
        anomaly_data: Dict[str, Any],
        severity: AlertSeverity
    ) -> str:
        """Generate alert message from anomaly data."""
        anomaly_count = anomaly_data.get('anomalies_detected', 0)
        total_windows = anomaly_data.get('total_windows', 0)
        
        if anomaly_count == 0:
            return "No anomalies detected"
        
        percentage = (anomaly_count / total_windows * 100) if total_windows > 0 else 0
        
        severity_text = {
            AlertSeverity.LOW: "Low severity",
            AlertSeverity.MEDIUM: "Medium severity",
            AlertSeverity.HIGH: "High severity",
            AlertSeverity.CRITICAL: "CRITICAL"
        }
        
        return (
            f"{severity_text[severity]} anomaly alert: "
            f"{anomaly_count} anomalies detected ({percentage:.1f}% of data)"
        )
    
    def _get_default_channels(
        self,
        severity: AlertSeverity
    ) -> List[NotificationChannel]:
        """Get default channels based on severity."""
        if severity == AlertSeverity.CRITICAL:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS]
        elif severity == AlertSeverity.HIGH:
            return [NotificationChannel.EMAIL, NotificationChannel.SLACK]
        elif severity == AlertSeverity.MEDIUM:
            return [NotificationChannel.SLACK]
        else:
            return [NotificationChannel.WEBHOOK]
    
    def _send_to_channels(
        self,
        alert: Dict[str, Any],
        channels: List[NotificationChannel]
    ) -> Dict[NotificationChannel, Dict[str, Any]]:
        """Send alert to multiple channels."""
        results = {}
        
        for channel in channels:
            # Check rate limit
            if self.enable_rate_limiting and not self._check_rate_limit(channel):
                logger.warning(f"Rate limit exceeded for {channel.value}")
                results[channel] = {'success': False, 'error': 'rate_limit_exceeded'}
                continue
            
            # Send notification
            result = self._send_to_channel(alert, channel)
            results[channel] = result
            
            # Update rate limit tracking
            if result['success'] and self.enable_rate_limiting:
                self._update_rate_limit(channel)
        
        return results
    
    def _send_to_channel(
        self,
        alert: Dict[str, Any],
        channel: NotificationChannel
    ) -> Dict[str, Any]:
        """Send alert to specific channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                return self._send_email(alert)
            elif channel == NotificationChannel.SLACK:
                return self._send_slack(alert)
            elif channel == NotificationChannel.WEBHOOK:
                return self._send_webhook(alert)
            elif channel == NotificationChannel.SMS:
                return self._send_sms(alert)
            elif channel == NotificationChannel.TEAMS:
                return self._send_teams(alert)
            else:
                return {'success': False, 'error': f'Unsupported channel: {channel}'}
        except Exception as e:
            logger.error(f"Failed to send to {channel.value}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_email(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification."""
        # In production, implement actual email sending
        logger.info(f"Sending email alert: {alert['id']}")
        self._stats['sent_notifications'] += 1
        return {'success': True, 'method': 'email', 'timestamp': datetime.now().isoformat()}
    
    def _send_slack(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification."""
        # In production, implement actual Slack integration
        logger.info(f"Sending Slack alert: {alert['id']}")
        self._stats['sent_notifications'] += 1
        return {'success': True, 'method': 'slack', 'timestamp': datetime.now().isoformat()}
    
    def _send_webhook(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification."""
        # In production, implement actual webhook sending
        logger.info(f"Sending webhook alert: {alert['id']}")
        self._stats['sent_notifications'] += 1
        return {'success': True, 'method': 'webhook', 'timestamp': datetime.now().isoformat()}
    
    def _send_sms(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS notification."""
        # In production, implement actual SMS sending
        logger.info(f"Sending SMS alert: {alert['id']}")
        self._stats['sent_notifications'] += 1
        return {'success': True, 'method': 'sms', 'timestamp': datetime.now().isoformat()}
    
    def _send_teams(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Send Microsoft Teams notification."""
        # In production, implement actual Teams integration
        logger.info(f"Sending Teams alert: {alert['id']}")
        self._stats['sent_notifications'] += 1
        return {'success': True, 'method': 'teams', 'timestamp': datetime.now().isoformat()}
    
    def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if rate limit allows sending."""
        limit_config = self._rate_limits[channel]
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Clean old entries
        limit_config['sent'] = [
            t for t in limit_config['sent']
            if t >= current_hour
        ]
        
        # Check limit
        return len(limit_config['sent']) < limit_config['max_per_hour']
    
    def _update_rate_limit(self, channel: NotificationChannel) -> None:
        """Update rate limit tracking."""
        self._rate_limits[channel]['sent'].append(datetime.now())
    
    def _is_duplicate_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert is duplicate."""
        # Simple duplicate detection - in production, use more sophisticated logic
        recent_alerts = list(self._alert_history)[-10:]
        for recent in recent_alerts:
            if (recent.get('type') == alert.get('type') and
                recent.get('severity') == alert.get('severity') and
                abs((datetime.fromisoformat(recent['timestamp']) - 
                     datetime.fromisoformat(alert['timestamp'])).total_seconds()) < 300):
                return True
        return False
    
    def _record_alert(
        self,
        alert: Dict[str, Any],
        results: Dict[NotificationChannel, Dict[str, Any]]
    ) -> None:
        """Record alert in history."""
        alert['notification_results'] = {
            ch.value: res for ch, res in results.items()
        }
        self._alert_history.append(alert)
        self._stats['total_alerts'] += 1
    
    def _flush_batch(self, channel: NotificationChannel) -> Dict[str, Any]:
        """Flush batch queue for channel."""
        batch = self._batch_queues[channel]
        if not batch:
            return {'status': 'empty', 'channel': channel.value}
        
        # Send batch
        result = self._send_to_channel({'alerts': batch, 'batch': True}, channel)
        
        # Clear queue
        self._batch_queues[channel] = []
        
        return {
            'status': 'sent',
            'channel': channel.value,
            'batch_size': len(batch),
            'result': result
        }
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"alert_{uuid.uuid4().hex[:8]}"
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load notification configuration."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_templates(self) -> Dict[str, str]:
        """Load notification templates."""
        return {
            'email_subject': "IoT Anomaly Alert: {severity}",
            'email_body': "Alert ID: {id}\nSeverity: {severity}\nMessage: {message}\nTimestamp: {timestamp}",
            'slack_message': ":warning: *{severity} Alert*\n{message}\n_Alert ID: {id}_",
            'sms_message': "{severity}: {message}"
        }
    
    def _validate_channel_config(
        self,
        channel: NotificationChannel,
        config: Dict[str, Any]
    ) -> bool:
        """Validate channel configuration."""
        required_fields = {
            NotificationChannel.EMAIL: ['smtp_host', 'smtp_port', 'from_email'],
            NotificationChannel.SLACK: ['webhook_url'],
            NotificationChannel.WEBHOOK: ['url'],
            NotificationChannel.SMS: ['api_key', 'from_number'],
            NotificationChannel.TEAMS: ['webhook_url']
        }
        
        if channel in required_fields:
            return all(field in config for field in required_fields[channel])
        return True
    
    def _test_channel(
        self,
        channel: NotificationChannel,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Test notification channel."""
        test_alert = {
            'id': 'test',
            'type': 'test',
            'message': 'Test notification',
            'timestamp': datetime.now().isoformat()
        }
        
        # In production, actually test the channel
        return {'success': True, 'tested_at': datetime.now().isoformat()}
    
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        data: Dict[str, Any]
    ) -> bool:
        """Evaluate rule condition."""
        # Simplified condition evaluation
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if field not in data:
            return False
        
        data_value = data[field]
        
        if operator == '>':
            return data_value > value
        elif operator == '<':
            return data_value < value
        elif operator == '==':
            return data_value == value
        elif operator == '>=':
            return data_value >= value
        elif operator == '<=':
            return data_value <= value
        
        return False
    
    def _execute_action(
        self,
        action: Dict[str, Any],
        data: Dict[str, Any]
    ) -> None:
        """Execute rule action."""
        action_type = action.get('type')
        
        if action_type == 'notify':
            channels = [NotificationChannel(c) for c in action.get('channels', [])]
            severity = AlertSeverity(action.get('severity', 'medium'))
            self.send_system_alert(
                alert_type='rule_triggered',
                message=action.get('message', 'Alert rule triggered'),
                details=data,
                severity=severity
            )