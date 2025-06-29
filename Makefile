.PHONY: test lint security integration setup

lint:
	ruff check .

security:
	bandit -r src

# Run lint, security, and tests
test:
	./scripts/test.sh

# Run only integration tests
integration:
	pytest -m integration

setup:
	./scripts/setup.sh
