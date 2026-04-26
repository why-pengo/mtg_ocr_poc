.PHONY: format lint test ci

format:
	black .
	isort .

lint:
	black --check .
	isort --check .
	flake8 .

test:
	.venv/bin/pytest

ci: lint test
