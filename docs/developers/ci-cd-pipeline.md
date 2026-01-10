# CI/CD Pipeline (Overview)

This page outlines CI/CD practices for DocMind AI.

## Targets

- Python 3.11 (project constraint: `>=3.11,<3.12`)
- Offline‑deterministic unit/integration tests (no network)
- Lint and format: Ruff (format + check); Ruff enforces pylint-equivalent rules via `PL*` selectors in `pyproject.toml`

## Suggested Steps

1) Lint and format

    ```bash
    uv run ruff format .
    uv run ruff check . --fix
    uv run pyright
    ```

2) Run tests (fast subsets on CI)

    ```bash
    uv run python scripts/run_tests.py --fast
    ```

3) Coverage (scoped to changed subsystems)

    ```bash
    uv run python scripts/run_tests.py --coverage
    ```

4) Docs validation (optional)

   - Grep for legacy terms: QueryPipeline (retrieval), alpha, rrf_k, LanceDB, SPLADE++, KGIndex
   - Ensure router_factory and server‑side fusion terminology present in updated docs

## Notes

- Tests should stub heavy integrations (Qdrant, LlamaIndex, Ollama) and enforce offline determinism.
- Avoid brittle string assertions in UI tests; prefer structural/behavioral assertions.
