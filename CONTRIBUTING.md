# Contributing to IDRAAK

## Development Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Code Style

- Use `ruff` for linting and formatting
- Type hints on all public functions
- Docstrings on public classes and functions

## Adding a New Language

1. Add the language to `configs/languages/default.yaml`
2. Add glossary entries to `glossaries/`
3. Update prompts if language-specific handling is needed
4. Add test cases

## Adding a New Provider

1. Implement the `TranslationProvider` or `LLMProvider` protocol
2. Add to `src/idraak/providers/`
3. Register in `__init__.py`
4. Add configuration options
5. Add tests with mocks

## Adding a New Agent

1. Inherit from `BaseAgent`
2. Define input/output schemas
3. Implement `run()` method
4. Add to the workflow orchestration
5. Add tests

## Adding a Drift Type

1. Add to `DriftType` enum in `schemas/drift.py`
2. Add comparison logic in `drift/comparator.py`
3. Add perturbation generator in `perturbations/engine.py`
4. Update evaluation to track the new type
5. Add tests
