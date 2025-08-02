# CI/CD Setup Guide

## Overview

This document provides comprehensive instructions for setting up CI/CD workflows for the IoT Anomaly Detection system. Due to GitHub App permission limitations, workflows must be set up manually.

## Workflow Files Location

All workflow templates are located in:
- `docs/workflows-templates/` - Basic templates (already copied to `.github/workflows/`)
- `.github/workflows/` - Active workflow files

## Available Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Validates code quality, runs tests, and performs security scans on every push and pull request.

**Features:**
- Multi-Python version testing (3.8-3.12)
- Code quality checks (ruff, black, mypy)
- Comprehensive test suite
- Security scanning
- Coverage reporting

**Triggers:**
- Push to `main` and `develop` branches
- Pull requests to `main`

**Key Steps:**
1. Environment setup
2. Dependency installation
3. Code quality checks
4. Unit tests
5. Integration tests
6. Security scans
7. Coverage reporting

### 2. Dependency Updates (`dependency-update.yml`)

**Purpose**: Automated dependency management with security scanning.

**Features:**
- Weekly dependency updates
- Security vulnerability scanning
- Automated testing of updates
- Auto-merge for patch updates (if configured)

**Triggers:**
- Schedule: Weekly on Mondays
- Manual trigger

### 3. Release Pipeline (`release.yml`)

**Purpose**: Automated release process with version management.

**Features:**
- Semantic version bumping
- Release notes generation
- Docker image building and publishing
- GitHub release creation

**Triggers:**
- Push to tags matching `v*`
- Manual trigger

### 4. Security Scanning (`security.yml`)

**Purpose**: Comprehensive security analysis.

**Features:**
- Static code analysis (Bandit, Semgrep)
- Dependency vulnerability scanning (Safety)
- Container security scanning (Trivy)
- Secret detection (TruffleHog)

**Triggers:**
- Push to main branches
- Pull requests
- Weekly scheduled scans

### 5. Performance Testing (`performance.yml`)

**Purpose**: Performance benchmarking and regression detection.

**Features:**
- Performance benchmarks
- Memory profiling
- Load testing
- Regression detection

**Triggers:**
- Push to main
- Weekly scheduled runs

## Manual Setup Instructions

### Step 1: Verify Workflow Files

Ensure all workflow files are in place:

```bash
ls -la .github/workflows/
```

Expected files:
- `ci.yml`
- `dependency-update.yml`
- `release.yml`
- `security.yml`
- `performance.yml`

### Step 2: Configure Repository Secrets

Navigate to: **Settings → Secrets and variables → Actions**

#### Required Secrets:

1. **`CODECOV_TOKEN`**
   - Get from [codecov.io](https://codecov.io)
   - Used for coverage reporting

2. **`DOCKER_USERNAME`** and **`DOCKER_PASSWORD`**
   - Docker Hub credentials
   - Used for container image publishing

3. **`PYPI_API_TOKEN`** (if publishing to PyPI)
   - PyPI API token
   - Used for package publishing

4. **`SLACK_WEBHOOK_URL`** (optional)
   - Slack webhook for notifications
   - Used for build notifications

#### Optional Secrets:

1. **`SONAR_TOKEN`** - SonarQube integration
2. **`SNYK_TOKEN`** - Snyk security scanning
3. **`GITHUB_TOKEN`** - Automatically provided by GitHub

### Step 3: Configure Repository Settings

#### Branch Protection Rules

Navigate to: **Settings → Branches → Add rule**

For the `main` branch:

1. **Branch name pattern**: `main`
2. **Protect matching branches**:
   - ✅ Require pull request reviews before merging
   - ✅ Dismiss stale PR reviews when new commits are pushed
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require signed commits (recommended)
   - ✅ Include administrators

3. **Required status checks**:
   - CI Tests (Python 3.11)
   - Security Scan
   - Code Quality
   - Coverage Check

#### Environment Protection Rules

For production deployments:

1. **Environment name**: `production`
2. **Protection rules**:
   - Required reviewers: 2
   - Wait timer: 5 minutes
   - Deployment branches: `main` only

### Step 4: Enable GitHub Features

Navigate to: **Settings → Security & analysis**

Enable:
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning alerts (if GitHub Advanced Security available)
- ✅ Secret scanning alerts (if GitHub Advanced Security available)

### Step 5: Configure Dependabot

Dependabot configuration is already in place at `.github/dependabot.yml`.

Features:
- Weekly Python dependency updates
- Docker base image updates
- GitHub Actions updates
- Automatic security updates

### Step 6: Test Workflow Setup

1. **Create a test branch**:
   ```bash
   git checkout -b test-ci-setup
   echo "# Test CI" >> test-ci.md
   git add test-ci.md
   git commit -m "test: verify CI setup"
   git push origin test-ci-setup
   ```

2. **Create a pull request** and verify:
   - CI workflow triggers
   - All checks pass
   - Status checks appear in PR

3. **Check Actions tab** for workflow execution details

## Workflow Configuration

### Environment Variables

Common environment variables used across workflows:

```yaml
env:
  PYTHON_VERSION: 3.11
  NODE_VERSION: 18
  DOCKER_BUILDKIT: 1
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
```

### Caching Strategy

Workflows use caching to improve performance:

```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

### Matrix Testing

CI workflow tests multiple Python versions:

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, 3.10, 3.11, 3.12]
    os: [ubuntu-latest]
  fail-fast: false
```

## Notifications Setup

### Slack Integration

1. Create Slack app with incoming webhook
2. Add webhook URL to repository secrets as `SLACK_WEBHOOK_URL`
3. Notifications will be sent for:
   - Build failures
   - Security alerts
   - Release deployments

### Email Notifications

GitHub automatically sends email notifications for:
- Failed workflows (to commit author)
- Security alerts
- Dependency updates

### Status Badges

Add badges to README.md:

```markdown
[![CI](https://github.com/danieleschmidt/iot-anomaly-detector-timeseries/actions/workflows/ci.yml/badge.svg)](https://github.com/danieleschmidt/iot-anomaly-detector-timeseries/actions/workflows/ci.yml)
[![Security](https://github.com/danieleschmidt/iot-anomaly-detector-timeseries/actions/workflows/security.yml/badge.svg)](https://github.com/danieleschmidt/iot-anomaly-detector-timeseries/actions/workflows/security.yml)
[![Coverage](https://codecov.io/gh/danieleschmidt/iot-anomaly-detector-timeseries/branch/main/graph/badge.svg)](https://codecov.io/gh/danieleschmidt/iot-anomaly-detector-timeseries)
```

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check branch protection rules
   - Verify file location (`.github/workflows/`)
   - Check YAML syntax

2. **Authentication failures**
   - Verify repository secrets
   - Check token permissions
   - Ensure secrets are not expired

3. **Test failures**
   - Check test environment setup
   - Verify dependencies are installed
   - Review test isolation

4. **Security scan failures**
   - Review security findings
   - Update vulnerable dependencies
   - Add security exceptions if needed

### Debug Commands

```bash
# Validate workflow YAML
cd .github/workflows
for file in *.yml; do
  echo "Validating $file"
  cat "$file" | python -c "import yaml, sys; yaml.safe_load(sys.stdin)"
done

# Check repository secrets
gh secret list

# View workflow runs
gh run list

# Download workflow logs
gh run download <run-id>
```

## Best Practices

### Workflow Design

1. **Fast Feedback**
   - Run fast tests first
   - Fail fast on obvious issues
   - Use parallel jobs where possible

2. **Security**
   - Don't log sensitive data
   - Use least-privilege tokens
   - Validate external inputs

3. **Reliability**
   - Handle flaky tests
   - Use appropriate timeouts
   - Implement retry mechanisms

### Maintenance

1. **Regular Reviews**
   - Monthly workflow performance review
   - Quarterly security updates
   - Annual workflow optimization

2. **Documentation**
   - Keep this guide updated
   - Document custom actions
   - Maintain troubleshooting notes

3. **Monitoring**
   - Track workflow success rates
   - Monitor execution times
   - Alert on repeated failures

## Integration with External Services

### Code Coverage (Codecov)

1. Sign up at [codecov.io](https://codecov.io)
2. Install GitHub app
3. Add `CODECOV_TOKEN` to repository secrets
4. Coverage reports are automatically uploaded

### Container Registry (GitHub Container Registry)

Container images are automatically built and published to `ghcr.io` on releases.

### Deployment Targets

Configure deployment environments:
- **Development**: Auto-deploy on push to `develop`
- **Staging**: Auto-deploy on push to `main`
- **Production**: Manual approval required

## Monitoring and Metrics

### Workflow Metrics

Track:
- Build success rate
- Average build time
- Test coverage trends
- Security scan results

### Alerting

Set up alerts for:
- Consecutive build failures
- Security vulnerabilities
- Performance regressions
- Deployment failures

## Advanced Configuration

### Custom Actions

Repository includes custom actions in `.github/actions/`:
- Setup Python environment
- Security scanning
- Deployment utilities

### Reusable Workflows

Create reusable workflows for common patterns:
- Database testing
- Integration testing
- Security scanning

### Matrix Strategies

Use matrices for:
- Multiple Python versions
- Different operating systems
- Various dependency versions

## Support and Resources

### Internal Resources
- Repository discussions
- Team Slack channels
- Internal documentation

### External Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)

### Getting Help

1. Check existing issues and discussions
2. Review workflow logs and error messages
3. Consult team documentation
4. Reach out to DevOps team
5. Create support ticket if needed