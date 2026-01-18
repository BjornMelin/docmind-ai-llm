# CI/CD Pipeline (Overview)

This page outlines CI/CD practices for DocMind AI.

## Targets

- Python 3.13.11 (project constraint: `>=3.13,<3.14`)
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

## Troubleshooting (AppTest UI timeouts)

If CI fails with Streamlit AppTest timeouts (for example: `AppTest script run
timed out`), reproduce and profile locally:

```bash
# Run just the AppTest UI integration subset
uv run pytest tests/integration/ui -vv

# Identify slow tests
uv run pytest tests/integration/ui --durations=20 --durations-min=0.5

# Force a higher AppTest timeout (useful on slow runners / under coverage)
CI=1 TEST_TIMEOUT=40 uv run pytest tests/integration/ui -q
```

Notes:

- Timeouts are centralized in `tests/helpers/apptest_utils.py`
  (`apptest_timeout_sec()`); avoid per-test ad-hoc values.
- The suite pre-warms AppTest once in `tests/integration/conftest.py`.
- UI tests stub the provider badge health check in `tests/integration/ui/conftest.py`
  to avoid optional GraphRAG adapter discovery during UI renders.

1) Coverage (scoped to changed subsystems)

    ```bash
    uv run python scripts/run_tests.py --coverage
    ```

2) Docs validation (optional)

   - Grep for legacy terms: QueryPipeline (retrieval), alpha, rrf_k, LanceDB, SPLADE++, KGIndex
   - Ensure router_factory and server‑side fusion terminology present in updated docs

## Notes

- Tests should stub heavy integrations (Qdrant, LlamaIndex, Ollama) and enforce offline determinism.
- Avoid brittle string assertions in UI tests; prefer structural/behavioral assertions.
