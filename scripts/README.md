# Repository scripts

DocMind keeps a small set of fail-closed utilities around the standard Python
toolchain. Pytest, Coverage.py, Ruff, and Pyright own quality enforcement; the
repository does not maintain a parallel quality-orchestration framework.

## Canonical verification

Run the complete local gate before shipping:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ruff check src --config ruff-core.toml
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q \
  --cov=src \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json \
  --cov-fail-under=80 \
  --junitxml=junit.xml
uv run python scripts/check_links.py
uv run python scripts/verify_structural_parity.py
uv run python scripts/validate_schemas.py
```

Pytest owns test selection, coverage enforcement, and report generation. The coverage command writes `coverage.xml`, `coverage.json`, `htmlcov/`, and `junit.xml`, then fails below 80 percent.

Use these native lanes:

```bash
uv run pytest tests/unit -q --no-cov
uv run pytest tests/integration -q --no-cov
uv run pytest tests/unit tests/integration -q --no-cov
uv run --no-sync pytest -m requires_gpu --no-cov
uv run pytest tests/unit/processing/test_parser_contract.py -vv --no-cov
```

`check_coverage.py` is an optional report helper. CI passes the 80 percent floor and each output format directly to Pytest.

## Runtime and release proof

- `benchmark_parsing.py` runs the schema-3 parser benchmark in isolated worker
  processes. Benchmark a clean commit and retain the JSON alongside the exact
  hardware, model cache, corpus, and command used:

  ```bash
  uv run python scripts/benchmark_parsing.py \
    --generate-minimal-fixtures \
    --output cache/benchmarks/parsing/results.json
  ```

- `parser_health.py --check` verifies parser imports and the app-owned Docling
  layout bundle. Offline fixture tests own RapidOCR inference validation.
- `test_gpu.py --quick` validates an explicitly installed GPU profile.
- `container_health.py` is the Docker health-check owner.
- `check_release_contract.py` verifies release version parity across the root
  project metadata, root package lock entry, Release Please manifest, and newest
  released changelog heading. Run `uv lock --check` beside it.

## Data and service utilities

- `backup.py`: create a complete local recovery point with uploads and exact
  Qdrant snapshots by default, or prune verified recovery points. Diagnostic
  omission flags create incomplete captures that never enter retention. The
  manifest's `databases.cache_db` and `databases.chat_db` entries record each
  native copy's backup-relative `path`, `size_bytes`, and `sha256`. Both complete
  and incomplete creates exit 0, so automation must require `complete=true` and
  an empty `warnings` list. Follow the operations guide for restore.
- `cleanup_collections.py`: inspect deployment-owned orphan Qdrant generations
  after stopping every DocMind reader and writer. It is dry-run by default;
  `--delete` is an explicit second step.
- `qdrant_schema.py`: inspect or rebuild an empty canonical Qdrant collection.
- `qdrant_fusion_smoke.py`: prove client support for native RRF and DBSF.
- `start_qdrant_local.sh`: start the supported local Qdrant service.
- `run_ingestion_demo.py`: exercise the ingestion boundary with local inputs.
- `demo_metrics_console.py`: render local metrics for development diagnostics.

## Documentation and repository checks

- `check_links.py`: validate internal documentation links.
- `verify_structural_parity.py`: compare documented and implemented structure.
- `validate_schemas.py`: validate the supported BEIR leaderboard schema.
- `analyze_github_reviews.py`: summarize exported review JSON without requiring
  repository write access.

Use `uv run python <script> --help` for script-specific options. Never treat a
missing artifact, baseline, dependency, or subprocess result as a successful
quality signal.
