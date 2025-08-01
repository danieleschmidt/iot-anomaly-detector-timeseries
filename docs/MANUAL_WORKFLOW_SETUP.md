# Manual Workflow Setup Required

## ⚠️ GitHub App Permission Limitation

The Terragon SDLC implementation is complete, but GitHub Actions workflow files could not be pushed automatically due to GitHub App lacking `workflows` permission. This is a security limitation.

## 🚀 Quick Setup Instructions

### Step 1: Create Workflow Files

Copy the following workflow files to `.github/workflows/` in your repository:

#### 1. CI/CD Pipeline (`ci.yml`)
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

permissions:
  contents: read
  security-events: write

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with ruff
      run: ruff check .
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
```

#### 2. Security Scanning (`security.yml`)
```yaml
name: Security Scan

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

permissions:
  contents: read
  security-events: write

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety semgrep
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Run Bandit Security Scanner
      run: bandit -r src/ -f txt
      continue-on-error: true
    
    - name: Run Safety Check
      run: safety check
      continue-on-error: true
    
    - name: Run Semgrep
      run: semgrep --config=auto src/
      continue-on-error: true
```

#### 3. Dependency Updates (`dependency-update.yml`)
```yaml
name: Dependency Update

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install pip-tools
      run: pip install pip-tools
    
    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'deps: update Python dependencies'
        title: 'Automated dependency update'
        body: |
          This PR updates Python dependencies to their latest versions.
          
          Please review the changes and ensure all tests pass before merging.
        branch: automated-dependency-update
```

### Step 2: Configure Repository Secrets

Go to **Settings → Secrets and variables → Actions** and add:

1. **`CODECOV_TOKEN`** - Get from [codecov.io](https://codecov.io)
2. **`DOCKER_USERNAME`** and **`DOCKER_PASSWORD`** - For container publishing
3. Optional: `SLACK_WEBHOOK_URL`, `SONAR_TOKEN`, `SNYK_TOKEN`

### Step 3: Enable Repository Features

Go to **Settings → Security & analysis** and enable:
- ✅ Dependency graph
- ✅ Dependabot alerts  
- ✅ Dependabot security updates
- ✅ Code scanning alerts (if available)
- ✅ Secret scanning alerts (if available)

### Step 4: Set Up Branch Protection

Go to **Settings → Branches → Add rule** for `main` branch:
- ✅ Require pull request reviews before merging
- ✅ Dismiss stale PR reviews when new commits are pushed
- ✅ Require status checks to pass before merging
- ✅ Include administrators

Required status checks:
- CI Tests (Python 3.11)
- Security Scan

### Step 5: Verify Setup

1. Create a test branch and PR
2. Verify workflows trigger and pass
3. Check that all integrations work

## 📋 Complete Workflow Files

For the complete, production-ready workflow files, see the existing templates in `docs/workflows-templates/` and refer to the comprehensive setup guide in `docs/workflows/CI_CD_SETUP.md`.

## ✅ Validation

Once setup is complete, run:
```bash
# Check repository health
python scripts/automation/repository_health_check.py

# Collect SDLC metrics
python scripts/automation/metrics_collector.py
```

## 🆘 Support

- Check `docs/workflows/CI_CD_SETUP.md` for detailed instructions
- Review `docs/runbooks/INCIDENT_RESPONSE.md` for troubleshooting
- Use automation scripts for health checks and metrics

The SDLC implementation is complete and ready - just needs these workflow files deployed manually!