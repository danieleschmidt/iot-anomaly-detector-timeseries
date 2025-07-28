# Workflow Requirements

## Overview

This document outlines the manual workflow setup requirements for the SDLC automation system.

## Required GitHub Actions Workflows

### 1. Continuous Integration (CI)
- **File**: `.github/workflows/ci.yml`
- **Triggers**: Push to main, pull requests
- **Requirements**: Python testing, linting, security scans
- **Template**: Available in `docs/workflows-templates/ci.yml`

### 2. Dependency Updates
- **File**: `.github/workflows/dependency-update.yml`
- **Triggers**: Schedule (weekly)
- **Requirements**: Automated dependency updates via Dependabot
- **Template**: Available in `docs/workflows-templates/dependency-update.yml`

### 3. Release Management
- **File**: `.github/workflows/release.yml`
- **Triggers**: Tag creation
- **Requirements**: Automated releases with changelog generation
- **Template**: Available in `docs/workflows-templates/release.yml`

## Manual Setup Required

Due to security restrictions, these workflows must be manually created by repository administrators:

1. Copy templates from `docs/workflows-templates/` to `.github/workflows/`
2. Configure repository secrets for deployment
3. Enable branch protection rules
4. Configure Dependabot settings

## Branch Protection Configuration

Enable the following branch protection rules for `main`:

- Require pull request reviews
- Dismiss stale reviews when new commits are pushed
- Require status checks to pass before merge
- Require branches to be up to date before merge
- Include administrators in restrictions

## Required Repository Secrets

Configure these secrets in repository settings:

- `CODECOV_TOKEN`: For coverage reporting
- `DOCKER_REGISTRY_TOKEN`: For container publishing
- `RELEASE_TOKEN`: For automated releases

For detailed setup instructions, see `docs/SETUP_REQUIRED.md`.