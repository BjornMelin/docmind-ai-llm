# Configure DocMind tests

This reference explains DocMind’s test settings, dependency profiles, collection toggles, and parser evidence. Use it to configure the smallest environment that exercises the boundary you changed.

## Follow the configuration owners

These files define the active test contract:

- `pyproject.toml`: dependency profiles, Pytest settings, markers, and coverage floor
- `tests/conftest.py`: global isolation, service stubs, and `requires_llama` behavior
- `tests/fixtures/test_settings.py`: unit, integration, and system settings factories
- `tests/e2e/conftest.py`: end-to-end (E2E) collection toggle
- `tests/system/conftest.py`: system collection toggle
- `scripts/run_tests.py`: tiered test runner

Update the owner before documenting a new profile, marker, or toggle.

## Select a settings fixture

The fixture module exposes three current settings classes:

| Class | Factory | Purpose |
| --- | --- | --- |
| `MockDocMindSettings` | `create_test_settings()` | Unit boundaries with local paths, CPU defaults, short retries, and disabled caches |
| `IntegrationTestSettings` | `create_integration_settings()` | Cross-component wiring with controlled model and service seams |
| `SystemTestSettings` | `create_system_settings()` | Production settings model with no test-specific field overrides |

Use the factory that matches the test boundary:

```python
from tests.fixtures.test_settings import create_test_settings


def test_local_override(tmp_path):
    settings = create_test_settings(data_dir=tmp_path)

    assert settings.data_dir == tmp_path
```

The shared `integration_settings` and `system_settings` fixtures call the corresponding factories. A system test does not require a graphics processing unit (GPU) unless it carries `requires_gpu`.

## Keep tests isolated from local settings

The global autouse fixture removes `DOCMIND_*` variables before each test. It also resets the application settings singleton with dotenv loading disabled.

Set a runtime override with `monkeypatch` inside the test before constructing settings. Do not depend on a developer `.env` file.

The fixture classes use these prefixes when you instantiate them directly:

- `MockDocMindSettings`: `DOCMIND_TEST_`
- `IntegrationTestSettings`: `DOCMIND_INTEGRATION_`
- `SystemTestSettings`: `DOCMIND_`

Nested fields use `__`, such as `DOCMIND_TEST_VLLM__MAX_TOKENS`.

## Install the matching dependency profile

`pyproject.toml` and `uv.lock` define every supported profile:

| Boundary | Command | Adds |
| --- | --- | --- |
| Baseline development | `uv sync --frozen` | Development tools and official CPU-only PyTorch wheels |
| GPU | `uv sync --frozen --no-group cpu --extra gpu` | CUDA 12.8 PyTorch, torchvision, and CuPy; sparse FastEmbed remains CPU-based |
| Apple acceleration | `uv sync --frozen --extra apple` | spaCy Apple operations on supported Apple Silicon hosts |
| Visual reranking | `uv sync --frozen --extra multimodal` | ColPali reranking dependencies |
| Observability | `uv sync --frozen --extra observability` | LlamaIndex OpenTelemetry integration |
| Evaluation | `uv sync --frozen --extra eval` | BEIR evaluation dependencies |
| Searchable PDF export | `uv sync --frozen --extra searchable-pdf` | OCRmyPDF |

The baseline includes `llama-index-core`, its built-in property graph store, selected LlamaIndex storage and large language model adapters, direct FastEmbed, and Transformers SigLIP support. There are no `graph` or `llama` extras and no `--extras` test lane.

The `multimodal` extra adds ColPali only. It does not select another image-embedding backbone.

## Configure collection-time toggles

Set collection toggles before starting Pytest:

| Variable | Effect |
| --- | --- |
| `DOCMIND_RUN_E2E=1` | Runs `tests/e2e`; the directory is skipped otherwise |
| `DOCMIND_RUN_SYSTEM=1` | Runs `tests/system`; the directory is skipped otherwise |
| `DOCMIND_QDRANT_SCHEMA_SMOKE=1` | Runs the optional named-vector schema smoke test |
| `REQUIRE_REAL_LLAMA=1` | Fails a `requires_llama` test when `llama_index.core` is unavailable |
| `TEST_TIMEOUT=20` | Overrides the Streamlit AppTest timeout in seconds |

Continuous integration (CI) sets `REQUIRE_REAL_LLAMA=1`. Local runs skip `requires_llama` tests only when the required core package is unavailable.

## Use active markers

The current suite uses these project markers:

- `unit`: isolated domain behavior
- `integration`: cross-component behavior
- `e2e`: application workflow
- `system`: explicit runtime or local-service validation
- `requires_gpu`: GPU hardware required
- `requires_llama`: real `llama_index.core` boundary required
- `retrieval`: retrieval feature selection

Register a new marker in `pyproject.toml` or `tests/conftest.py` before using it.

## Run supported test lanes

Run a focused owner first:

```bash
uv run pytest tests/unit/processing/test_parser_contract.py -vv
```

Run unit and integration tiers through the runner:

```bash
uv run python scripts/run_tests.py --unit
uv run python scripts/run_tests.py --integration
uv run python scripts/run_tests.py --fast
uv run python scripts/run_tests.py --coverage
```

Run opt-in collections explicitly:

```bash
DOCMIND_RUN_E2E=1 uv run pytest tests/e2e -vv
DOCMIND_RUN_SYSTEM=1 uv run pytest tests/system -vv
```

Run the GPU-marked tests only after installing the GPU profile:

```bash
uv run python scripts/run_tests.py --gpu
```

## Validate the Qdrant boundary

Start the loopback-only Qdrant container before running its live schema smoke test:

```bash
./scripts/start_qdrant_local.sh
uv run python scripts/qdrant_schema.py check
DOCMIND_QDRANT_SCHEMA_SMOKE=1 \
  uv run pytest \
  tests/integration/retrieval/test_qdrant_named_vectors_schema.py \
  -vv
```

Unit tests use a controlled Qdrant client seam. Do not convert a missing live service into a production fallback.

## Validate the parser boundary

The local PDF parser uses Docling, pypdfium2, RapidOCR, and ONNX Runtime. Plain text and Markdown use the direct text loader. Fixed parser identity fields are validation literals, not backend selectors.

Prefetch and verify the pinned parser models before PDF integration or benchmark runs:

```bash
uv run python tools/models/pull.py \
  --parser-defaults \
  --rapidocr-cache-dir cache/models
uv run python scripts/parser_health.py --check
```

Run the parser contract and model-integrity tests:

```bash
uv run pytest \
  tests/unit/processing/test_parser_contract.py \
  tests/unit/processing/test_model_artifacts.py \
  tests/unit/processing/test_parser_health.py \
  -vv
```

OCRmyPDF is an optional searchable-PDF export utility. It is not a parser backend.

## Regenerate parser benchmark evidence

The benchmark harness emits schema 3 and validates eight generated fixtures across three isolated repetitions:

```bash
uv run python scripts/benchmark_parsing.py \
  --generate-minimal-fixtures \
  --repeat 3 \
  --output docs/benchmarks/parser-runtime-validation.json
```

Treat an artifact with another schema version as stale. The harness records `network_egress: NOT_MEASURED`, so it does not prove zero network traffic.

Do not publish unverified benchmark claims from another machine or dependency state. Record the repository identity and dirty state already included in the artifact.

## Preserve fixture ownership

Keep cross-domain defaults in `tests/conftest.py` or `tests/shared_fixtures.py`. Keep domain fixtures beside their consumers.

Patch the consumer seam before constructing Streamlit AppTest or a service object. Use `tmp_path`, deterministic inputs, and `monkeypatch` instead of real network calls or sleeps.
