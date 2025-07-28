# Manual Setup Requirements

## Overview

This document lists all manual setup steps required to complete the SDLC automation.

## GitHub Repository Settings

### 1. Branch Protection Rules
Navigate to: Settings > Branches > Add rule

**For `main` branch:**
- ✅ Require pull request reviews before merging
- ✅ Dismiss stale PR reviews when new commits are pushed
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

### 2. Repository Secrets
Navigate to: Settings > Secrets and variables > Actions

**Required Secrets:**
- `CODECOV_TOKEN`: Get from [codecov.io](https://codecov.io)
- `DOCKER_REGISTRY_TOKEN`: Registry authentication token
- `RELEASE_TOKEN`: GitHub token with repo permissions

### 3. Repository Topics
Navigate to: Repository main page > ⚙️ Settings (gear icon)

**Add Topics:**
- anomaly-detection
- iot
- machine-learning
- python
- tensorflow
- monitoring

## GitHub Actions Workflows

### Manual Deployment Steps

1. **Copy Workflow Templates**
   ```bash
   cp docs/workflows-templates/*.yml .github/workflows/
   ```

2. **Verify Workflow Syntax**
   - GitHub will automatically validate YAML syntax
   - Check Actions tab for any errors

3. **Enable Workflows**
   - Navigate to Actions tab
   - Enable workflows if prompted

## Dependabot Configuration

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

## Security Configuration

### 1. Enable Vulnerability Alerts
Navigate to: Settings > Security & analysis
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates

### 2. Code Scanning
Enable GitHub Advanced Security (if available):
- ✅ Code scanning alerts
- ✅ Secret scanning alerts

## Integration Setup

### 1. CodeCov Integration
1. Sign up at [codecov.io](https://codecov.io)
2. Connect GitHub repository
3. Copy token to repository secrets

### 2. Pre-commit.ci Integration
1. Visit [pre-commit.ci](https://pre-commit.ci)
2. Enable for repository
3. Configuration already exists in `.pre-commit-config.yaml`

## Verification Checklist

After completing manual setup:

- [ ] Branch protection rules active
- [ ] All repository secrets configured
- [ ] Workflows copied and enabled
- [ ] Dependabot configuration active
- [ ] Security features enabled
- [ ] Integrations connected and working
- [ ] First PR/merge triggers CI successfully

## Support

For setup assistance:
- **Documentation**: Check existing docs in `/docs`
- **Issues**: Create GitHub issue with `setup` label
- **Contact**: dev@terragonlabs.com