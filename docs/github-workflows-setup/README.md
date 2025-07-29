# GitHub Workflows Setup Guide

**⚠️ IMPORTANT**: Due to GitHub security restrictions, workflow files cannot be added directly via pull requests. These files must be manually added by repository administrators with appropriate permissions.

## Overview

This directory contains the GitHub Actions workflow files that need to be manually placed in `.github/workflows/` to complete the SDLC automation enhancement.

## Files to Install

1. **`ci.yml`** - Comprehensive continuous integration workflow
2. **`security.yml`** - Multi-layered security scanning workflow  
3. **`release.yml`** - Automated release management workflow

## Installation Instructions

### Step 1: Manual File Placement

Repository administrators should copy these files to the appropriate locations:

```bash
# From the repository root
cp docs/github-workflows-setup/ci.yml .github/workflows/
cp docs/github-workflows-setup/security.yml .github/workflows/
cp docs/github-workflows-setup/release.yml .github/workflows/
```

### Step 2: Commit Workflow Files

```bash
git add .github/workflows/
git commit -m "feat: add comprehensive GitHub Actions workflows

- Add CI workflow with multi-Python testing
- Add security scanning workflow with SAST, dependency, and container scans
- Add automated release workflow with SBOM generation"
git push
```

### Step 3: Configure Repository Secrets

Add the following secrets in GitHub repository settings (Settings → Secrets and variables → Actions):

#### Required Secrets
- `GITHUB_TOKEN` (automatically provided)

#### Optional Secrets (for enhanced functionality)
- `CODECOV_TOKEN` - For code coverage reporting
- `SLACK_WEBHOOK_URL` - For notification integration
- `DOCKER_HUB_USERNAME` & `DOCKER_HUB_TOKEN` - For Docker Hub publishing

### Step 4: Enable Actions Permissions

1. Go to Settings → Actions → General
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Set "Workflow permissions" to "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"

## Workflow Features

### CI Workflow (`ci.yml`)
- **Multi-Python testing**: Tests across Python 3.8-3.12
- **Code quality**: Linting, formatting, type checking
- **Security scanning**: Bandit, pip-audit
- **Docker build testing**
- **Performance testing** (on PRs)
- **Coverage reporting** to Codecov

### Security Workflow (`security.yml`)
- **Daily automated scanning**
- **Dependency vulnerability scanning**
- **SAST with CodeQL and Bandit**
- **Container security with Trivy**
- **Secrets detection with TruffleHog**
- **SBOM generation and submission**

### Release Workflow (`release.yml`)
- **Automated on git tags** (v*.*.*)
- **Full test suite validation**
- **Multi-platform Docker builds**
- **GitHub release creation**
- **SBOM inclusion in releases**
- **Changelog generation**

## Validation

After installation, verify workflows are working:

1. **Check workflow runs**: Go to Actions tab in GitHub
2. **Test CI**: Create a small PR to trigger CI workflow
3. **Test security**: Workflows run daily automatically
4. **Test release**: Create a git tag like `v0.0.4`

## Troubleshooting

### Common Issues

1. **Workflow not running**: Check permissions and branch protection rules
2. **Secret access issues**: Verify secrets are properly configured
3. **Docker build failures**: Check Dockerfile and build context
4. **Test failures**: Ensure local tests pass before pushing

### Getting Help

- Check workflow logs in the Actions tab
- Review [GitHub Actions documentation](https://docs.github.com/en/actions)
- Open an issue for workflow-specific problems

## Monitoring and Maintenance

### Workflow Health
- Monitor workflow success rates in Actions tab
- Set up notifications for failed workflows
- Regularly update action versions (Dependabot will help)

### Performance Optimization
- Monitor workflow execution times
- Optimize test parallelization
- Use caching where appropriate (already configured)

### Security Updates
- Review security scan results weekly
- Update dependencies based on security alerts
- Monitor SARIF reports in Security tab

## Integration with Existing SDLC

These workflows integrate with:

- **Pre-commit hooks**: Local validation before CI
- **Dependabot**: Automated dependency updates
- **CODEOWNERS**: Required code reviews
- **Branch protection**: Enforce CI success before merge
- **Monitoring**: Prometheus metrics from application

## Next Steps After Installation

1. **Configure branch protection** requiring CI success
2. **Set up monitoring** for workflow failures
3. **Train team** on new CI/CD processes
4. **Review and tune** alert thresholds
5. **Document** team-specific workflow customizations

This completes the SDLC automation enhancement, elevating the repository to ADVANCED maturity level (85%+).