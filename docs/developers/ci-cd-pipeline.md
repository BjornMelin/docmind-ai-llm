# CI/CD Pipeline (Concise)

## Overview

A simple GitHub Actions workflow runs linting and a stable smoke test suite using uv and pytest. The pipeline targets Python 3.11 by default and also validates on 3.10. Python 3.12 is not supported.

## Supported Python Versions

- Default: 3.11
- Also tested: 3.10
- Not supported: 3.12

## GitHub Actions Workflow

Reference implementation: `.github/workflows/pr-validation.yml`

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [ main ]

jobs:
  unit:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.10"]

    steps:
      - name: Check out
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv (with cache)
        id: setup-uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          cache-dependency-glob: "**/pyproject.toml"

      - name: Install dependencies
        run: uv sync --group test --group dev

      - name: Ruff format (check only)
        run: uv run ruff format --check .

      - name: Ruff lint
        run: uv run ruff check .

      - name: Pylint (src + unit tests)
        run: uv run pylint -j 0 -sn --rcfile=pyproject.toml src tests/unit

      - name: Run unit tests with coverage
        run: |
          uv run pytest tests/unit -m unit --cov=src --cov-report=xml:coverage.xml --cov-report=term -q
```

Notes

- Use ruff in non-fixing mode in CI to avoid write diffs.
- Keep CI fast by running a stable smoke suite; expand as reliability improves.
- For nightly jobs, you may run broader unit/integration tests and collect coverage.

## Local Pre-commit (Recommended)

```bash
ruff format . && ruff check . --fix
uv run pytest tests/unit/config/test_settings.py -v
```

## Rationale

- Aligns with uv-only package management and boundary-first testing.
- Keeps the pipeline simple and maintainable.
