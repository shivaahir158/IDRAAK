.PHONY: install test lint demo generate evaluate run clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install in development mode
	pip install -e ".[dev]"

test:  ## Run all tests
	python3 -m pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	python3 -m pytest tests/ -v --tb=short --cov=src/idraak --cov-report=term-missing

lint:  ## Run linters
	python3 -m ruff check src/ tests/

format:  ## Format code
	python3 -m ruff format src/ tests/

demo:  ## Run quick demo (no API keys needed)
	python3 -m idraak.cli demo

generate:  ## Generate full dataset (300 requirements)
	python3 -m idraak.cli generate-dataset --n 300

generate-sample:  ## Generate sample dataset (10 requirements)
	python3 -m idraak.cli generate-dataset --n 10 --output data/sample/requirements.jsonl

evaluate:  ## Run evaluation on default dataset
	python3 -m idraak.cli evaluate

run:  ## Run complete end-to-end pipeline
	python3 -m idraak.cli run

clean:  ## Clean generated files
	rm -rf artifacts/cache/ reports/*.png reports/*.pdf reports/*.csv reports/*.md reports/*.tex
	rm -rf data/interim/* data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
