#!/usr/bin/env bash
# Run pytest test suite with environment variables loaded from .env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

cd "$PROJECT_DIR"

# Load .env into the shell so all API keys and config are available to tests
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    echo "Loaded .env from $PROJECT_DIR"
else
    echo "WARNING: .env not found at $PROJECT_DIR/.env — some tests may be skipped"
fi

# Override REDIS_HOST for local runs (Docker .env points to 'redis' service name)
export REDIS_HOST="${REDIS_HOST_LOCAL:-localhost}"

# Ensure project root is on PYTHONPATH for app module imports
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Activate virtualenv if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Activated virtualenv: $VIRTUAL_ENV"
fi

echo ""
echo "Running tests..."
echo "========================================"

# Pass all script arguments to pytest (e.g. -v, -m llm, -k test_name)
pytest tests/ "$@"
