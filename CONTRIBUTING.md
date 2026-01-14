# Contributing to DocMind AI

Thank you for your interest in contributing to DocMind AI! This guide outlines the workflow and standards for this project.

## Workflow

1. **Branching**: Create a feature branch from `main`.

    ```bash
    git checkout -b feature/your-feature-name
    ```

2. **Development**: Implement your changes following the [Architecture Guide](docs/developers/system-architecture.md).
3. **Testing**: Ensure all tests pass.

    ```bash
    uv run python scripts/run_tests.py --fast
    ```

4. **Linting**: Run `ruff` and `pyright` to ensure code quality.

    ```bash
    uv run ruff check .
    uv run pyright
    ```

5. **Pull Request**: Open a PR against `main`. Ensure your description clearly states the problem and solution.

## Tooling

We use `uv` for dependency management and `ruff` for linting/formatting.

- **Check Python code quality**: `uv run ruff check .`
- **Format code**: `uv run ruff format .`
- **Type check**: `uv run pyright`
- **Run tests**: `uv run python scripts/run_tests.py`
- **Validate documentation links**: `python scripts/check_links.py docs/`

## Documentation Standards

- Always update the relevant ADR (Architecture Decision Record) in `docs/developers/adrs` for significant architectural changes.
- Ensure all internal links in `docs/` are valid. Use `scripts/check_links.py` to verify.
- Maintain the directory structure manifest in `system-architecture.md`.

## Quality Gates

Every PR must pass the following automated checks:

- 0 linting errors (Ruff/Pyright).
- 100% test pass rate.
- 100% internal documentation link resolution (external links not validated).
- 1:1 structural parity between code and architecture docs.
