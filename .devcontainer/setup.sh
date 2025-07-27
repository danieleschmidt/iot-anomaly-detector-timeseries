#!/bin/bash

# Development container setup script
set -e

echo "🚀 Setting up development environment..."

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    make \
    tree \
    htop \
    jq \
    unzip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p data/processed data/models logs

# Set up Git configuration (if not already configured)
if ! git config user.name > /dev/null 2>&1; then
    echo "⚙️  Setting up Git configuration..."
    git config --global user.name "Developer"
    git config --global user.email "developer@example.com"
    git config --global init.defaultBranch main
fi

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
fi

# Install additional tools
echo "🛠️  Installing additional development tools..."

# Install Node.js for frontend development (if needed)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Set up shell aliases for development
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias grep='grep --color=auto'

# Project-specific aliases
alias test='pytest -v'
alias lint='ruff check .'
alias format='black .'
alias coverage='pytest --cov=src --cov-report=html'
alias serve='python -m src.model_serving_api'
alias train='python -m src.train_autoencoder'
alias detect='python -m src.anomaly_detector'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcb='docker-compose build'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
alias gd='git diff'
EOF

# Make scripts executable
chmod +x scripts/*.sh

# Display development tips
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Quick start commands:"
echo "  make setup     - Install dependencies"
echo "  make test      - Run tests"
echo "  make lint      - Run linting"
echo "  make format    - Format code"
echo "  make coverage  - Run tests with coverage"
echo "  make build     - Build Docker image"
echo "  make serve     - Start API server"
echo ""
echo "🔧 Development workflow:"
echo "  1. Create feature branch: git checkout -b feature/your-feature"
echo "  2. Make changes and test: make test"
echo "  3. Format and lint: make format && make lint"
echo "  4. Commit changes: git commit -m 'feat: your feature'"
echo "  5. Push and create PR: git push origin feature/your-feature"
echo ""
echo "📚 Documentation:"
echo "  - Architecture: docs/ARCHITECTURE.md"
echo "  - API docs: docs/api/"
echo "  - Contributing: CONTRIBUTING.md"
echo ""