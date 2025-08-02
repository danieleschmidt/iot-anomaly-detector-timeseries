# Incident Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to incidents in the IoT Anomaly Detection system.

## Incident Severity Levels

### P0 - Critical (Service Down)
- **Response Time**: 15 minutes
- **Escalation**: Immediate page to on-call engineer
- **Examples**: System completely down, data loss, security breach

### P1 - High (Major Impact)
- **Response Time**: 1 hour
- **Escalation**: Slack notification + email
- **Examples**: High error rate, performance degradation, partial outage

### P2 - Medium (Minor Impact)
- **Response Time**: 4 hours
- **Escalation**: Email notification
- **Examples**: Non-critical feature issues, warning thresholds exceeded

### P3 - Low (Minimal Impact)
- **Response Time**: 1 business day
- **Escalation**: Ticket creation
- **Examples**: Documentation issues, minor bugs

## Common Incidents

### 1. System Completely Down

**Symptoms:**
- All API endpoints returning 5xx errors
- Grafana showing no metrics
- Health check endpoint unreachable

**Immediate Actions:**
1. Check if it's a planned maintenance (check calendar)
2. Verify the issue affects all users
3. Check infrastructure status (AWS, GCP, etc.)

**Investigation Steps:**
```bash
# Check container status
docker-compose ps

# View recent logs
docker-compose logs --tail=100 app

# Check system resources
docker stats
df -h
```

**Resolution Steps:**
1. Restart the application service
   ```bash
   docker-compose restart app
   ```

2. If restart fails, check for:
   - Database connectivity
   - Missing environment variables
   - Resource exhaustion

3. Roll back to last known good version if needed
   ```bash
   git log --oneline -10
   git checkout <last-good-commit>
   docker-compose up --build -d
   ```

**Post-Resolution:**
- Monitor for stability (15-30 minutes)
- Update incident status
- Schedule post-mortem

### 2. High API Error Rate

**Symptoms:**
- Error rate > 10% for 5+ minutes
- Increased response times
- Customer complaints

**Investigation Steps:**
```bash
# Check error logs
docker-compose logs app | grep ERROR | tail -50

# Check API metrics
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])

# Check database connections
docker-compose exec postgres psql -U user -d anomaly_db -c "SELECT count(*) FROM pg_stat_activity;"
```

**Resolution Steps:**
1. Identify error patterns in logs
2. Check for recent deployments
3. Verify database health
4. Scale application if needed
   ```bash
   docker-compose up --scale app=3
   ```

**Common Causes:**
- Database connection pool exhaustion
- Memory leaks
- Invalid model artifacts
- Network connectivity issues

### 3. Model Inference Failures

**Symptoms:**
- Prediction endpoints returning errors
- High reconstruction errors
- Model loading failures

**Investigation Steps:**
```bash
# Check model files
ls -la saved_models/
du -sh saved_models/*

# Check model loading logs
docker-compose logs app | grep -i "model"

# Verify model API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 25.0, "pressure": 1013.25, "humidity": 60.0, "vibration": 0.1}'
```

**Resolution Steps:**
1. Verify model file integrity
2. Check available memory/disk space
3. Restart model service
4. Fall back to previous model version if needed

### 4. Database Connectivity Issues

**Symptoms:**
- Connection timeout errors
- Database queries failing
- Slow response times

**Investigation Steps:**
```bash
# Check database container
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec app python -c "
import psycopg2
try:
    conn = psycopg2.connect(host='postgres', database='anomaly_db', user='user', password='password')
    print('Connection successful')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
"
```

**Resolution Steps:**
1. Restart database container
   ```bash
   docker-compose restart postgres
   ```

2. Check for database locks
   ```sql
   SELECT * FROM pg_locks WHERE NOT granted;
   ```

3. Verify database disk space
4. Check for connection pool issues

### 5. Memory/Resource Exhaustion

**Symptoms:**
- Out of memory errors
- Slow performance
- Container restarts

**Investigation Steps:**
```bash
# Check memory usage
free -h
docker stats --no-stream

# Check disk space
df -h

# Check for memory leaks
docker-compose exec app python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Resolution Steps:**
1. Restart affected services
2. Increase resource limits
3. Optimize batch sizes
4. Clear temporary files
   ```bash
   docker system prune -f
   ```

### 6. Data Pipeline Failures

**Symptoms:**
- No new data being processed
- Data validation errors
- Missing or corrupted files

**Investigation Steps:**
```bash
# Check data directories
ls -la data/raw/ data/processed/

# Check processing logs
docker-compose logs app | grep -i "data"

# Verify data sources
curl -I http://data-source-api/health
```

**Resolution Steps:**
1. Verify data source availability
2. Check file permissions
3. Restart data processing service
4. Manual data recovery if needed

## Escalation Procedures

### When to Escalate

1. **Immediate Escalation (P0):**
   - Unable to restore service within 30 minutes
   - Data loss confirmed
   - Security incident suspected

2. **Escalation After Initial Response (P1):**
   - No progress after 2 hours
   - Customer impact increasing
   - Multiple systems affected

### Escalation Contacts

1. **Technical Lead**: +1-555-0101
2. **SRE Manager**: +1-555-0102
3. **VP Engineering**: +1-555-0103

### Communication Channels

- **Internal**: #incidents Slack channel
- **External**: Status page updates
- **Customer**: Support ticket updates

## Communication Templates

### Initial Response (within 15 minutes)
```
🚨 INCIDENT ALERT - P[X] 

Summary: [Brief description]
Impact: [Who/what is affected]
Status: Investigating
ETA: [Expected resolution time]
Next Update: [Time]

Incident Commander: [Name]
```

### Status Update (every 30 minutes for P0, hourly for P1)
```
📊 INCIDENT UPDATE - P[X]

Progress: [What has been done]
Current Status: [Current state]
Next Steps: [What's being done next]
ETA: [Updated estimate]
Next Update: [Time]
```

### Resolution Notice
```
✅ INCIDENT RESOLVED - P[X]

Resolution: [What fixed it]
Root Cause: [If known]
Duration: [Total downtime]
Next Steps: [Post-mortem, etc.]

Thank you for your patience.
```

## Tools and Resources

### Monitoring
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093

### Logs
```bash
# Application logs
docker-compose logs -f app

# Database logs
docker-compose logs -f postgres

# All services
docker-compose logs -f
```

### Useful Commands
```bash
# Service health
docker-compose ps
docker-compose top

# Resource usage
docker stats
df -h
free -h

# Network connectivity
curl -v http://localhost:8000/health
ping database-host
```

### Configuration Files
- `docker-compose.yml` - Service configuration
- `config/monitoring/prometheus.yml` - Metrics configuration
- `config/monitoring/alert_rules.yml` - Alert definitions

## Post-Incident Procedures

### Immediate (within 24 hours)
1. Ensure system stability
2. Update monitoring if needed
3. Document lessons learned
4. Update this runbook if necessary

### Short-term (within 1 week)
1. Conduct post-mortem meeting
2. Create action items for prevention
3. Update alerts and monitoring
4. Share learnings with team

### Long-term (within 1 month)
1. Implement preventive measures
2. Update documentation
3. Review and test procedures
4. Update training materials

## Contact Information

### On-Call Schedule
- **Primary**: Check PagerDuty schedule
- **Secondary**: Check PagerDuty schedule
- **Escalation**: Technical Lead

### Vendor Contacts
- **Cloud Provider**: [Support number]
- **Database**: [Support portal]
- **Monitoring**: [Support email]

### Internal Contacts
- **Security Team**: security@company.com
- **DevOps Team**: devops@company.com
- **Product Team**: product@company.com

## Training and Drills

### Monthly Drills
- Simulated outage response
- Escalation procedure testing
- Communication channel testing

### Quarterly Reviews
- Runbook updates
- Process improvements
- Tool evaluations

### Annual Training
- New team member onboarding
- Advanced incident response
- Cross-team coordination

## Appendix

### Useful Queries

**Prometheus Queries:**
```promql
# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Latency P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Memory usage
memory_usage_bytes / 1024 / 1024
```

**Log Searches:**
```bash
# Recent errors
docker-compose logs --since=30m app | grep ERROR

# Database connection issues
docker-compose logs app | grep -i "connection\|database"

# Memory issues
docker-compose logs app | grep -i "memory\|oom"
```

### Recovery Scripts
Location: `scripts/recovery/`
- `restart_services.sh` - Safe service restart
- `health_check.sh` - Comprehensive health check
- `backup_recovery.sh` - Data recovery procedures