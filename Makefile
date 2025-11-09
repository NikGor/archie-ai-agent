# Makefile for Archie AI Agent

.PHONY: help install run dev chat voice test clean docker-build docker-run docker-stop lint format check black mypy ruff

# Default target
help:
	@echo "Archie AI Agent - Available commands:"
	@echo ""
	@echo "  install      - Install dependencies using Poetry"
	@echo "  run          - Run the FastAPI server"
	@echo "  dev          - Run in development mode with auto-reload"
	@echo "  chat         - Run console chat interface"
	@echo "  voice        - Run voice assistant"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run with Docker Compose"
	@echo "  docker-stop  - Stop Docker containers"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  format       - Format code with black"
	@echo "  lint         - Run all linters (ruff, mypy)"
	@echo "  check        - Run all checks (format check + lint)"
	@echo "  clean-lines  - Remove extra blank lines in code"
	@echo "  fix          - Auto-fix all issues and clean up code"
	@echo "  black        - Run black formatter"
	@echo "  mypy         - Run mypy type checker"
	@echo "  ruff         - Run ruff linter"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed successfully!"

# Run FastAPI server
run:
	@echo "Starting Archie AI Agent server..."
	poetry run uvicorn main:app --host 0.0.0.0 --port 8000

# Run in development mode with auto-reload
dev:
	@echo "Starting development server with auto-reload..."
	poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run console chat
chat:
	@echo "Starting console chat interface..."
	poetry run python -m app.chat

# Run voice assistant
voice:
	@echo "Starting voice assistant..."
	poetry run python -m app.voice

# Run tests (placeholder for future tests)
test:
	@echo "Running tests..."
	@echo "No tests defined yet."

# Clean cache and temporary files
clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup completed!"

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker-compose build

docker-run:
	@echo "Starting with Docker Compose..."
	docker-compose up -d
	@echo "Server is running at http://localhost:8001"

docker-stop:
	@echo "Stopping Docker containers..."
	docker-compose down

# Setup development environment
setup: install
	@echo "Development environment setup completed!"
	@echo "Run 'make run' to start the server"

# Show server status
status:
	@echo "Checking server status..."
	@curl -s http://localhost:8000/docs > /dev/null && echo "✅ Server is running at http://localhost:8000" || echo "❌ Server is not running"

# Test API with curl
api-test:
	@echo "Testing API..."
	@curl -X POST "http://localhost:8000/chat" \
		-H "Content-Type: application/json" \
		-d '{"response_format": "ui_answer", "input": "Hello from Makefile!", "conversation_id": "makefile_test", "model": "gpt-4"}' \
		2>/dev/null | python -m json.tool || echo "❌ API test failed - make sure server is running"

# Code formatting and linting commands
format:
	@echo "🎨 Formatting code with black..."
	poetry run black .
	@echo "✅ Code formatting completed!"

black:
	@echo "🎨 Running black formatter..."
	poetry run black .

black-check:
	@echo "🔍 Checking code formatting with black..."
	poetry run black --check --diff .

mypy:
	@echo "🔍 Running mypy type checker..."
	poetry run mypy .

ruff:
	@echo "🔍 Running ruff linter..."
	poetry run ruff check .

ruff-fix:
	@echo "🔧 Running ruff with auto-fix..."
	poetry run ruff check --fix .

# Remove extra blank lines and fix spacing issues
clean-lines:
	@echo "🧹 Cleaning up extra blank lines and organizing imports..."
	poetry run ruff check --select E303,E304,E305,E301,E302,I --fix .
	@echo "✅ Blank lines and imports cleaned up!"

lint: ruff mypy
	@echo "✅ All linting checks completed!"

check: black-check lint
	@echo "✅ All code quality checks completed!"

# Fix all auto-fixable issues
fix: ruff-fix format clean-lines
	@echo "🔧 Auto-fixed all issues, formatted code, and cleaned up blank lines!"
