# Incident Response Runbooks

This document provides structured incident response procedures for the IoT Anomaly Detector system to ensure rapid resolution and minimize impact.

## Quick Reference

**Emergency Contacts:**
- **On-call Engineer**: +1-XXX-XXX-XXXX
- **Security Team**: security@terragonlabs.com  
- **Infrastructure Team**: infra@terragonlabs.com
- **Management Escalation**: incidents@terragonlabs.com

**Status Page**: https://status.terragonlabs.com
**Incident Dashboard**: https://monitoring.terragonlabs.com/incidents

---

## Severity Classification

### P0 - Critical (< 15 min response)
- System completely down
- Data breach or security incident
- Financial/legal liability

### P1 - High (< 1 hour response)
- Major feature degradation
- Performance severely impacted
- Multiple customers affected

### P2 - Medium (< 4 hours response)
- Minor feature issues
- Single customer impact
- Non-critical component failure

### P3 - Low (< 24 hours response)
- Documentation issues
- Enhancement requests
- Cosmetic problems

---

## Runbook 1: API Service Down

### Symptoms
- HTTP 5xx errors from API endpoints
- Health check failures
- No response from model serving API

### Immediate Actions (0-5 minutes)
```bash
# Check service status
kubectl get pods -n anomaly-detector
docker-compose ps

# Check logs for errors
kubectl logs -f deployment/anomaly-detector-api -n anomaly-detector
docker-compose logs api

# Check resource usage
kubectl top pods -n anomaly-detector
docker stats
```

### Diagnosis Steps
1. **Check Infrastructure**
   ```bash
   # Database connectivity
   psql -h postgres -d anomaly_db -U user -c "SELECT 1;"
   
   # Redis connectivity  
   redis-cli -h redis ping
   
   # Disk space
   df -h
   
   # Memory usage
   free -h
   ```

2. **Review Recent Changes**
   - Check recent deployments
   - Review configuration changes
   - Examine dependency updates

3. **Check External Dependencies**
   - Database connection pools
   - External API rate limits
   - Network connectivity

### Resolution Actions
```bash
# Quick restart (if no data loss risk)
kubectl rollout restart deployment/anomaly-detector-api
docker-compose restart api

# Scale up replicas
kubectl scale deployment/anomaly-detector-api --replicas=3

# Rollback to previous version
kubectl rollout undo deployment/anomaly-detector-api

# Emergency maintenance mode
kubectl patch service/anomaly-detector-api -p '{"spec":{"selector":{"app":"maintenance"}}}'
```

### Post-Incident
- Document root cause
- Update monitoring thresholds
- Schedule follow-up review

---

## Runbook 2: High Memory Usage / OOM Kills

### Symptoms  
- Containers being OOM killed
- Gradual memory increase
- Slow response times

### Immediate Actions
```bash
# Check memory usage patterns
kubectl top pods -n anomaly-detector --sort-by=memory
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check for memory leaks
python -m src.performance_monitor_cli --memory-profile

# Scale up memory limits
kubectl patch deployment/anomaly-detector-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"memory":"2Gi"}}}]}}}}'
```

### Diagnosis Steps
1. **Profile Memory Usage**
   ```bash
   # Generate memory dump
   py-spy dump --pid $(pgrep -f "model_serving_api")
   
   # Check for memory-intensive operations
   grep -r "large_data" src/
   ```

2. **Review Data Processing**
   - Check batch sizes
   - Review caching strategies  
   - Examine model loading patterns

### Resolution Actions
```bash
# Implement memory limits
echo "Configure resource limits in deployment"

# Enable memory monitoring
echo "Add memory alerts to monitoring"

# Optimize code
echo "Review and optimize memory-intensive operations"
```

---

## Runbook 3: Security Incident

### Symptoms
- Unauthorized access attempts
- Suspicious API calls
- Security scanner alerts

### Immediate Actions (DO NOT DELAY)
```bash
# ISOLATE AFFECTED SYSTEMS
kubectl patch deployment/anomaly-detector-api -p '{"spec":{"replicas":0}}'

# PRESERVE EVIDENCE
kubectl logs deployment/anomaly-detector-api > incident-logs-$(date +%Y%m%d-%H%M%S).txt

# BLOCK SUSPICIOUS IPs
# Add to firewall/security groups immediately
```

### Notification Chain
1. **Immediate (< 5 minutes)**
   - Security team notification
   - Incident commander assignment
   - Management notification

2. **Follow-up (< 30 minutes)**
   - Legal team notification (if data involved)
   - Customer notification preparation
   - External authorities (if required)

### Investigation Steps
```bash
# Audit logs analysis
grep -E "(admin|root|sudo)" /var/log/auth.log

# Check for data exfiltration
netstat -an | grep :443
ss -tuln | grep :443

# Database audit
SELECT * FROM audit_log WHERE timestamp > NOW() - INTERVAL '1 hour';
```

### Containment Actions
- Rotate all credentials
- Revoke API keys
- Update security rules
- Apply emergency patches

---

## Runbook 4: Model Performance Degradation

### Symptoms
- High false positive/negative rates
- Accuracy metrics declining
- Drift detection alerts

### Immediate Actions
```bash
# Check model metrics
python -m src.evaluate_model --model-path saved_models/autoencoder.h5 --quick-check

# Review recent data patterns
python -m src.data_drift_detector --window-size 1000 --alert-threshold 0.1

# Fallback to previous model
cp saved_models/autoencoder_backup.h5 saved_models/autoencoder.h5
```

### Diagnosis Steps
1. **Data Quality Check**
   ```bash
   python -m src.data_validator data/raw/sensor_data.csv --comprehensive
   ```

2. **Model Drift Analysis**
   ```bash
   python -m src.model_explainability --drift-analysis --output drift_report.json
   ```

3. **Performance Benchmarking**
   ```bash
   python -m benchmarks.performance_benchmarks --compare-baseline
   ```

### Resolution Actions
- Retrain model with recent data
- Adjust detection thresholds
- Update feature engineering
- Schedule model refresh

---

## Runbook 5: Database Connection Issues

### Symptoms
- Connection timeouts
- "Too many connections" errors  
- Slow query performance

### Immediate Actions
```bash
# Check connection pool
psql -h postgres -d anomaly_db -c "SELECT * FROM pg_stat_activity;"

# Kill long-running queries
psql -h postgres -d anomaly_db -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';"

# Check database locks
psql -h postgres -d anomaly_db -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

### Resolution Actions
```bash
# Increase connection limits
echo "max_connections = 200" >> postgresql.conf

# Optimize queries
EXPLAIN ANALYZE SELECT * FROM sensor_data WHERE timestamp > NOW() - INTERVAL '1 hour';

# Setup connection pooling
echo "Configure PgBouncer or similar connection pooler"
```

---

## Runbook 6: Container/Kubernetes Issues

### Symptoms
- Pods stuck in pending state
- ImagePullBackOff errors
- Resource quota exceeded

### Immediate Actions
```bash
# Check pod status
kubectl describe pod -n anomaly-detector

# Check resource quotas
kubectl describe quota -n anomaly-detector

# Check node resources
kubectl describe nodes
```

### Resolution Actions
```bash
# Clear failed pods
kubectl delete pods --field-selector=status.phase=Failed -n anomaly-detector

# Restart problematic deployments
kubectl rollout restart deployment/anomaly-detector-api -n anomaly-detector

# Scale cluster if needed
kubectl scale nodes --replicas=3
```

---

## Monitoring and Alerting Integration

### Prometheus Queries for Alerts
```promql
# High memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.8

# API error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05

# Model prediction latency
histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m])) > 1.0
```

### Log Analysis Patterns
```bash
# Error patterns
grep -E "(ERROR|CRITICAL|FATAL)" /var/log/app.log | tail -100

# Performance issues
grep -E "(slow|timeout|latency)" /var/log/app.log | tail -100

# Security events
grep -E "(auth|login|unauthorized)" /var/log/app.log | tail -100
```

---

## Post-Incident Process

### Documentation Requirements
1. **Incident Timeline**
   - First detection time
   - Response actions taken
   - Resolution time
   - Customer impact duration

2. **Root Cause Analysis**
   - Technical root cause
   - Contributing factors
   - Why detection/response could be improved

3. **Action Items**
   - Immediate fixes applied
   - Long-term improvements needed
   - Process changes required

### Follow-up Actions
- Schedule incident review meeting
- Update runbooks based on learnings
- Improve monitoring/alerting
- Conduct team training if needed

---

## Team Training

### Required Knowledge
- Container orchestration basics
- Database troubleshooting
- Python application debugging
- Security incident response
- Communication protocols

### Training Schedule
- Monthly incident response drills
- Quarterly runbook reviews
- Annual security incident simulation

---

## Contact Information

**Technical Escalation:**
- L1 Support: support@terragonlabs.com
- L2 Engineering: engineering@terragonlabs.com  
- L3 Architecture: architecture@terragonlabs.com

**Management Escalation:**
- Engineering Manager: em@terragonlabs.com
- VP Engineering: vpe@terragonlabs.com
- CTO: cto@terragonlabs.com

**External Contacts:**
- Cloud Provider Support: [Account-specific]
- Security Vendor: [Vendor-specific]
- Legal Counselor: legal@terragonlabs.com

---

*This document should be reviewed and updated quarterly or after any major incident.*