# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.0.3   | ✅ Fully supported |
| 0.0.2   | ⚠️ Critical fixes only |
| < 0.0.2 | ❌ End of life     |

## Reporting a Vulnerability

### Security Contact

For security vulnerabilities, please **DO NOT** create a public GitHub issue. Instead:

1. **Email**: Send details to `security@terragonlabs.com`
2. **Subject**: Use `[SECURITY] IoT Anomaly Detection - Vulnerability Report`
3. **Response Time**: We aim to acknowledge within 24 hours

### What to Include

Please provide:

- **Description**: Clear description of the vulnerability
- **Impact**: Potential security impact and affected components
- **Reproduction**: Step-by-step reproduction instructions
- **Environment**: System details where vulnerability was found
- **Proof of Concept**: Code, screenshots, or evidence (if safe to share)

### Vulnerability Assessment Process

1. **Initial Review** (24-48 hours)
   - Acknowledgment of report
   - Initial severity assessment
   - Assignment to security team

2. **Investigation** (3-7 days)
   - Detailed analysis and validation
   - Impact assessment and risk scoring
   - Development of mitigation strategy

3. **Resolution** (varies by severity)
   - **Critical**: 1-3 days
   - **High**: 3-7 days
   - **Medium**: 7-14 days
   - **Low**: 14-30 days

4. **Disclosure**
   - Private notification to reporter
   - Public security advisory (if applicable)
   - CVE assignment for significant issues

## Security Best Practices

### For Users

- Always use the latest supported version
- Implement proper input validation
- Use secure communication channels (HTTPS/TLS)
- Regularly update dependencies
- Monitor for security advisories

### For Contributors

- Follow secure coding practices
- Run security scans before submission
- Never commit secrets or credentials
- Use parameterized queries for database access
- Implement proper error handling without information leakage

## Security Features

### Built-in Security Measures

1. **Input Validation**
   - Comprehensive data validation for all inputs
   - Sanitization of file paths and parameters
   - Type checking and bounds validation

2. **Secure Authentication**
   - JWT-based authentication for API endpoints
   - Secure token generation and validation
   - Configurable token expiration

3. **Data Protection**
   - Secure handling of sensitive data
   - No logging of sensitive information
   - Secure temporary file management

4. **Network Security**
   - Configurable host binding
   - HTTPS enforcement options
   - Request rate limiting capabilities

5. **Container Security**
   - Minimal base images
   - Non-root user execution
   - Resource limits and constraints

### Security Testing

We implement multiple layers of security testing:

- **Static Analysis**: Bandit security linting
- **Dependency Scanning**: Automated vulnerability detection
- **Container Scanning**: Docker image security analysis  
- **Penetration Testing**: Regular security assessments

## Threat Model

### Assets
- Machine learning models and training data
- API endpoints and authentication tokens
- Configuration and deployment secrets
- User data and sensor readings

### Threats
- **Data Poisoning**: Malicious training data injection
- **Model Theft**: Unauthorized model access or extraction
- **API Abuse**: Unauthorized access to detection endpoints
- **Container Escape**: Privilege escalation in containerized environments
- **Supply Chain**: Compromised dependencies or build process

### Mitigations
- Input validation and anomaly detection
- Access controls and authentication
- Container security and isolation
- Dependency scanning and management
- Secure CI/CD pipeline

## Compliance

### Standards Alignment

- **OWASP Top 10**: Protection against common web vulnerabilities
- **NIST Cybersecurity Framework**: Risk management and security controls
- **ISO 27001**: Information security management best practices
- **SOC 2**: Security, availability, and confidentiality controls

### Audit Trail

- All security-relevant actions are logged
- Immutable audit logs with integrity protection
- Regular security monitoring and alerting
- Compliance reporting capabilities

## Security Contact Information

- **Security Team**: security@terragonlabs.com
- **Bug Bounty**: Currently not available
- **PGP Key**: Available on request for sensitive communications

## Acknowledgments

We thank the security research community for responsible disclosure and continuous improvement of our security posture.

---

*Last updated: 2025-07-29*
*Next review: 2025-10-29*