# MTG OCR POC - Makefile
#
# This Makefile provides commands for common development tasks.
# Run `make help` to see all available commands.

.PHONY: help install format format-check lint check test test-verbose test-cov clean ci

# Default Python interpreter
VENV := .venv
VENV_BIN := $(VENV)/bin
PYTHON := $(VENV_BIN)/python3
PIP := $(VENV_BIN)/pip

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)MTG OCR POC$(NC)"
	@echo "==========="
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Environment Setup
# =============================================================================

install: ## Install dependencies into .venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed$(NC)"

install-ci: ## Install CI-only lightweight dependencies
	@echo "$(BLUE)Installing CI dependencies...$(NC)"
	$(PIP) install -r requirements-ci.txt
	@echo "$(GREEN)CI dependencies installed$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code with isort...$(NC)"
	$(VENV_BIN)/isort .
	@echo "$(BLUE)Formatting code with black...$(NC)"
	$(VENV_BIN)/black .
	@echo "$(GREEN)Code formatted$(NC)"

format-check: ## Check code formatting without making changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(VENV_BIN)/isort --check .
	$(VENV_BIN)/black --check .
	@echo "$(GREEN)Format check passed$(NC)"

lint: ## Run flake8 linter
	@echo "$(BLUE)Running flake8...$(NC)"
	$(VENV_BIN)/flake8 .
	@echo "$(GREEN)Lint passed$(NC)"

check: format-check lint ## Run all code quality checks (format + lint)
	@echo "$(GREEN)All checks passed$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(VENV_BIN)/pytest
	@echo "$(GREEN)Tests complete$(NC)"

test-verbose: ## Run tests with verbose output
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	$(VENV_BIN)/pytest -v --tb=long
	@echo "$(GREEN)Tests complete$(NC)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(VENV_BIN)/pytest --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Remove Python artifacts and cache files
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete$(NC)"

# =============================================================================
# Combined Commands
# =============================================================================

ci: check test ## Run all CI checks (format, lint, test)
	@echo "$(GREEN)All CI checks passed$(NC)"
