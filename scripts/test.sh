#!/usr/bin/env bash
set -euo pipefail

ruff check .
bandit -r src
pytest -q

