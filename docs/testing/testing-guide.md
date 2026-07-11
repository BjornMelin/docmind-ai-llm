# Test DocMind

This guide explains DocMind’s test tiers, supported commands, dependency profiles, and contribution rules. Use it to choose the smallest gate that proves a change before running the release gates.

## Test tiers

DocMind separates tests by boundary:

| Tier | Path | Purpose |
| --- | --- | --- |
| Unit | `tests/unit` | Domain logic with controlled dependencies |
| Integration | `tests/integration` | Cross-component behavior and Streamlit AppTest |
| End to end | `tests/e2e` | Application workflows through public surfaces |
| System | `tests/system` | Explicit local-service and runtime validation |
| Validation | `tests/validation` | Production-readiness contracts |

System tests are opt-in. Set `DOCMIND_RUN_SYSTEM=1` before running `tests/system`.

## Unit layout

`tests/unit` mirrors source ownership. High-signal domains include:

- `agents`: supervisor, role agents, tools, registry, and error recovery
- `config`: settings, providers, and integration binding
- `models`: Pydantic contracts and embedding models
- `nlp`: spaCy services and enrichment transforms
- `pages` and `ui`: page helpers and UI components
- `persistence`: snapshots, chat, artifacts, and checkpoints
- `processing`: parser, ingestion, OCR, page fidelity, and model integrity
- `prompting`: packaged templates and rendering
- `retrieval`: hybrid search, deduplication, reranking, GraphRAG, and SigLIP
- `scripts`: benchmark, container, model, package, and Qdrant operator commands
- `security` and `telemetry`: security boundaries and metadata-only events
- `utils`: shared storage, image, monitoring, and canonicalization helpers

See `tests/README.md` for the compact directory reference.

## Run focused tests

Start with the owner’s test file:

```bash
uv run pytest tests/unit/processing/test_parser_contract.py -vv
```

Run a domain when a change crosses several files:

```bash
uv run pytest tests/unit/processing -q
uv run pytest tests/unit/retrieval -q
```

Run an integration boundary after changing wiring:

```bash
uv run pytest tests/integration/test_ingestion_pipeline_pdf_images.py -vv
```

## Run the tiered test runner

`scripts/run_tests.py` supports these current lanes:

```bash
uv run python scripts/run_tests.py --unit
uv run python scripts/run_tests.py --integration
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py --gpu
uv run python scripts/run_tests.py --coverage
uv run python scripts/run_tests.py --validate-imports
```

The runner has no `--extras` lane. LlamaIndex core is a required dependency, not an optional test profile.

The default runner executes unit and integration tiers. System tests remain explicit.

## Test dependency profiles

The baseline profile includes `llama-index-core`, selected storage and large language model adapters, direct FastEmbed, and Transformers SigLIP support:

```bash
uv sync --frozen
```

GraphRAG storage tests use the required LlamaIndex core package:

```bash
uv run pytest --no-cov \
  tests/unit/retrieval/test_llama_index_adapter.py \
  tests/unit/retrieval/test_graph_rag_factory.py \
  tests/unit/retrieval/test_router_factory_contract.py \
  tests/integration/test_graph.py \
  tests/integration/test_graphrag_exports.py
```

The `requires_llama` marker means a test needs the real installed `llama_index.core` instead of the default router stub. It does not refer to a `llama` extra.

Local runs skip a `requires_llama` test when the required package is missing. Continuous integration (CI) sets `REQUIRE_REAL_LLAMA=1` so that missing core packages fail.

See `docs/testing/test-configuration.md` for the GPU, Apple, visual reranking, observability, evaluation, and searchable-PDF profiles.

## CPU and GPU tests

The default test environment uses official CPU-only PyTorch wheels:

```bash
uv sync --frozen
uv run python -c \
  "import torch; assert torch.version.cuda is None; print(torch.__version__)"
```

GPU testing is optional:

```bash
uv sync --frozen --no-group cpu --extra gpu
uv run python scripts/run_tests.py --gpu
```

Mark tests that require hardware with `requires_gpu`. A test that only exercises device selection should use a stub and remain in the CPU lane.

## Run opt-in tests

Run the E2E and system collections explicitly:

```bash
DOCMIND_RUN_E2E=1 uv run pytest tests/e2e -vv
DOCMIND_RUN_SYSTEM=1 uv run pytest tests/system -vv
```

The system tier does not require a GPU unless a test carries `requires_gpu`.

Start the loopback-only Qdrant container for its live schema smoke test:

```bash
./scripts/start_qdrant_local.sh
uv run python scripts/qdrant_schema.py check
DOCMIND_QDRANT_SCHEMA_SMOKE=1 \
  uv run pytest \
  tests/integration/retrieval/test_qdrant_named_vectors_schema.py \
  -vv
```

Do not hide a missing required service behind a unit-test fallback. Unit and integration tests should inject a controlled boundary instead.

## Validate parser evidence

Parser fixtures cover native text, physical pages, tables, formulas, adversarial text, scanned pages, skewed images, and multilingual text.

Regenerate the frozen benchmark:

```bash
uv run python scripts/benchmark_parsing.py \
  --generate-minimal-fixtures \
  --repeat 3 \
  --output docs/benchmarks/parser-runtime-validation.json
```

The harness records `network_egress: NOT_MEASURED`. Do not present the artifact as a network-capture result.

The current code emits benchmark schema 3. Treat a schema 2 artifact as stale, and regenerate it after the final implementation and dependency state is frozen.

## Validate the wheel

Build and inspect the wheel:

```bash
rm -rf build docmind_ai_llm.egg-info
uv build --wheel --clear
uv run python scripts/smoke_built_wheel.py
```

The smoke test validates wheel files and metadata, then installs the wheel with `--no-deps` in an isolated environment. It proves package contents, version import, and prompt-resource loading. It does not prove a full pip dependency installation.

Focused wheel-contract tests live under `tests/unit/scripts/test_smoke_built_wheel.py`.

## Run code and documentation gates

Run code gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run python scripts/run_tests.py --fast
```

Run documentation gates:

```bash
uv run python scripts/check_links.py
uv run python scripts/verify_structural_parity.py
npx markdownlint-cli \
  --disable MD013 MD033 MD041 \
  -- 'docs/**/*.md'
```

## Use markers deliberately

The current suite uses these project markers:

- `unit`, `integration`, `system`, and `e2e` for test tiers
- `requires_gpu` for hardware-dependent tests
- `requires_llama` for a real LlamaIndex core boundary
- `retrieval` for retrieval feature selection

Register a new marker in the Pytest configuration before using it.

## Write boundary-first tests

Follow these rules:

1. Patch the consumer seam, not an implementation detail several layers away
2. Use `tmp_path` for files, caches, databases, and environment persistence
3. Block real network access unless the test carries `requires_network`
4. Use deterministic data and avoid real sleeps
5. Assert public outcomes, persisted state, and emitted metadata
6. Keep model and service imports lazy where the production boundary is lazy
7. Restore global LlamaIndex, Streamlit, and environment state through fixtures
8. Keep unit tests under 5 seconds and integration tests under 30 seconds when practical

## Test Streamlit pages

For Streamlit AppTest:

- Reuse an existing AppTest fixture
- Change to a temporary working directory
- Patch service and coordinator boundaries before `AppTest.from_file`
- Use `tests/helpers/apptest_utils.py` for timeout configuration
- Avoid polling and sleeps
- Stub GraphRAG health unless the adapter is under test
- Assert rendered state and saved configuration through public UI behavior

## Manage fixtures

Keep only cross-domain defaults in `tests/conftest.py` and `tests/unit/conftest.py`. Put domain fixtures beside their consumers.

Do not import one domain’s `conftest.py` from another domain. Move a genuinely shared fixture into `tests/shared_fixtures.py` or `tests/fixtures`.

## Coverage and quality

Generate coverage through the runner:

```bash
uv run python scripts/run_tests.py --coverage
```

The configured line-coverage floor is 80 percent. Branch coverage is recorded only when the selected command enables it.

Do not weaken a gate to accommodate an unrelated failure. Record the existing failure, prove the scoped change independently, and route the failure to its owner.
