# CI/CD Pipeline Setup Guide

This document provides instructions for setting up the comprehensive CI/CD pipeline for the IoT Anomaly Detection system.

## Overview

The CI/CD pipeline includes:
- ✅ Code quality and security checks
- ✅ Comprehensive testing (unit, integration, performance)
- ✅ Docker builds and security scanning
- ✅ Automated releases and deployments
- ✅ Dependency management
- ✅ Monitoring and observability

## Quick Setup

### 1. Copy Workflow Files

Copy the workflow files from `docs/workflows-templates/` to `.github/workflows/`:

```bash
# Create .github/workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy all workflow files
cp docs/workflows-templates/*.yml .github/workflows/
```

### 2. Required Secrets and Variables

Set up the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):

#### Required Secrets:
- `PYPI_API_TOKEN` - For publishing to PyPI
- `TEST_PYPI_API_TOKEN` - For testing PyPI releases
- `DOCKERHUB_USERNAME` - Docker Hub username (optional)
- `DOCKERHUB_TOKEN` - Docker Hub access token (optional)
- `SNYK_TOKEN` - Snyk security scanning token (optional)

#### Optional Secrets for Enhanced Features:
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `CODECOV_TOKEN` - For enhanced code coverage reporting

### 3. Repository Settings

Configure the following repository settings:

#### Branch Protection Rules
1. Go to Settings → Branches
2. Add protection rule for `main` branch:
   - Require a pull request before merging
   - Require status checks to pass before merging
   - Required status checks:
     - `quality`
     - `test`
     - `build`
     - `docker`
   - Require branches to be up to date before merging
   - Require conversation resolution before merging

#### GitHub Pages (Optional)
1. Go to Settings → Pages
2. Set source to "GitHub Actions"
3. Documentation will be automatically deployed

### 4. Environment Setup

Create the following environments (Settings → Environments):

#### `production` Environment
- Add protection rules requiring manual approval
- Set environment secrets if different from repository secrets

## Workflow Details

### 1. Main CI/CD Pipeline (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Weekly security scans (Mondays at 2 AM)

**Jobs:**
1. **Quality Checks**: Linting, type checking, security scanning
2. **Testing**: Unit tests, integration tests, performance tests
3. **Build**: Package building and validation
4. **Docker**: Container builds and security scanning
5. **Security**: CodeQL analysis, dependency scanning
6. **Deploy**: Automated deployment to staging/production

### 2. Dependency Updates (`dependency-update.yml`)

**Triggers:**
- Weekly (Mondays at 4 AM)
- Manual dispatch

**Features:**
- Automatic dependency updates (patch, minor, major)
- Security vulnerability scanning
- Automated PR creation
- Test validation of updated dependencies

### 3. Release Pipeline (`release.yml`)

**Triggers:**
- Tag pushes (`v*`)
- Manual dispatch with version selection

**Features:**
- Automated version bumping
- Package building and publishing
- Docker image creation and publishing
- GitHub release creation with notes
- Post-release tasks

## Customization

### Modifying Workflows

1. **Test Configuration**: Adjust test parameters in `pyproject.toml`
2. **Security Thresholds**: Modify vulnerability thresholds in workflow files
3. **Deployment Targets**: Update deployment steps for your infrastructure
4. **Notification Channels**: Add Slack, Teams, or email notifications

### Adding Custom Jobs

Example of adding a custom job to the main pipeline:

```yaml
custom-job:
  name: Custom Validation
  runs-on: ubuntu-latest
  needs: quality
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Custom validation
      run: |
        echo "Running custom validation..."
        # Add your custom commands here
```

## Monitoring Pipeline Health

### Workflow Status

Monitor workflow health through:
- GitHub Actions tab
- Repository badges (add to README)
- Slack/email notifications (if configured)

### Metrics and Insights

Track pipeline metrics:
- Success/failure rates
- Build times
- Test coverage trends
- Security vulnerability trends

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure GitHub App has necessary permissions
   - Check repository secrets are correctly set
   - Verify environment protection rules

2. **Test Failures**
   - Check test logs in Actions tab
   - Run tests locally: `make test`
   - Review test configuration in `pyproject.toml`

3. **Docker Build Issues**
   - Verify Dockerfile syntax
   - Check base image availability
   - Review Docker build logs

4. **Deployment Failures**
   - Verify deployment environment access
   - Check environment variables and secrets
   - Review deployment logs

### Debug Mode

Enable debug logging by setting repository secret:
```
ACTIONS_STEP_DEBUG = true
```

## Best Practices

### 1. Pull Request Workflow
1. Create feature branch from `develop`
2. Make changes and commit with conventional commits
3. Push branch and create pull request
4. Wait for all checks to pass
5. Request review and merge

### 2. Release Workflow
1. Ensure all features are merged to `main`
2. Update CHANGELOG.md
3. Create release tag or use manual dispatch
4. Monitor release pipeline
5. Verify deployment success

### 3. Security Practices
- Regularly review and update dependencies
- Monitor security scan results
- Address critical vulnerabilities immediately
- Keep secrets and tokens secure

### 4. Performance Monitoring
- Monitor build times and optimize if needed
- Track test execution times
- Review pipeline resource usage
- Optimize Docker builds with multi-stage builds

## Migration from Existing CI/CD

If migrating from another CI/CD system:

1. **Backup existing configuration**
2. **Map existing jobs to new workflows**
3. **Migrate secrets and environment variables**
4. **Test workflows on a branch first**
5. **Gradually transition services**

## Support and Maintenance

### Regular Tasks
- Weekly review of failed workflows
- Monthly security scan review
- Quarterly workflow optimization
- Annual dependency audit

### Updates
- Keep workflow actions updated
- Monitor for new security features
- Update base images regularly
- Review and update documentation

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

For questions or issues with the CI/CD setup, please create an issue in the repository or contact the development team.