# Develop DocMind AI

Use this handbook to set up the repository, find the canonical implementation owner, and run the required checks. Follow the linked references for subsystem details.

## Set up the repository

Create the locked environment and a local configuration file:

```bash
uv sync --frozen
cp .env.example .env
```

Prefetch the required local models before you enable offline mode:

```bash
uv run python tools/models/pull.py \
  --all \
  --cache_dir ./models_cache \
  --parser-defaults \
  --parser-cache-dir ./cache/models
```

Start Qdrant, verify its schema, and run the application:

```bash
./scripts/start_qdrant_local.sh
uv run python scripts/qdrant_schema.py check
uv run streamlit run app.py
```

The [developer setup guide](getting-started.md) covers CPU, NVIDIA, Apple Silicon, and external language-model server profiles.

## Change the canonical owner

Each runtime concern has one implementation owner:

| Concern | Canonical owner |
| --- | --- |
| Application settings | `src/config/settings.py` |
| Runtime initialization | `src/config/integrations.py` |
| Parsing | `src/processing/parsing/` |
| Ingestion input loading | `src/processing/ingestion_api.py` |
| Ingestion pipeline execution | `src/processing/ingestion_pipeline.py` |
| Retrieval composition | `src/retrieval/router_factory.py` |
| Agent orchestration | `src/agents/coordinator.py` and `src/agents/supervisor_graph.py` |
| Snapshots and artifacts | `src/persistence/` |
| OpenTelemetry | `src/telemetry/opentelemetry.py` |
| Local JSONL telemetry | `src/utils/telemetry.py` |

Update the owner and every affected caller in one change. Do not add parallel implementations, compatibility adapters, or hidden fallback paths.

## Keep configuration typed

Read application settings from the exported singleton:

```python
from src.config import settings

model = settings.effective_model
timeout_seconds = settings.agents.decision_timeout
```

Add subsystem settings to the appropriate Pydantic model in `src/config/settings.py`. Use `DOCMIND_` with `__` for nested environment fields. Keep environment reads inside `src/config/`, and avoid import-time file or network access.

The [configuration reference](configuration.md) documents precedence, endpoint policy, and supported environment variables.

## Preserve runtime boundaries

Apply these boundaries when you change application behavior:

- Route supported binary formats through `src/processing/parsing/`; only explicit text formats use the direct UTF-8 loader
- Build retrieval through `build_router_engine(...)`; the native LlamaIndex router owns semantic, hybrid, keyword, multimodal, and graph selection
- Call `MultiAgentCoordinator.process_query(...)` from application code; functions under `src/agents/tools/` are graph internals
- Persist binary artifacts as `ArtifactRef` values; do not store base64 payloads or absolute host paths
- Emit metadata-only logs and telemetry; never record prompts, document text, model output, credentials, or raw exception content

Read the [architecture overview](architecture-overview.md) before changing a boundary and update the owning specification or architecture decision record when the contract changes.

## Verify code changes

Run non-mutating format, lint, type, and test checks before handoff:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
```

Run the opt-in acceptance lanes when your change reaches those boundaries:

```bash
DOCMIND_RUN_E2E=1 uv run pytest tests/e2e -q --no-cov
DOCMIND_RUN_SYSTEM=1 \
  DOCMIND_QDRANT_SYSTEM_URL=http://127.0.0.1:6333 \
  uv run pytest tests/system/test_e2e_offline.py -q --no-cov
```

Use only registered project markers: `unit`, `integration`, `system`, `e2e`, `requires_gpu`, `requires_network`, `retrieval`, and `requires_llama`. The test suite registers `requires_llama` and controls its skip or fail behavior through `REQUIRE_REAL_LLAMA`.

The [testing guide](../testing/testing-guide.md) explains fixtures, Streamlit AppTest boundaries, coverage, and marker behavior.

## Validate documentation and release metadata

Run the documentation gates after you change Markdown, specifications, or architecture records:

```bash
uv run python scripts/check_links.py
uv run python scripts/verify_structural_parity.py
uv run python scripts/validate_schemas.py
npx --yes markdownlint-cli@0.47.0 \
  --disable MD013 MD033 MD041 -- 'docs/**/*.md'
```

Validate release-owned versions and the lockfile before release work:

```bash
uv run python scripts/check_release_contract.py
uv lock --check
```

Release Please owns version changes in `pyproject.toml`, `uv.lock`, `.release-please-manifest.json`, and `CHANGELOG.md`. The [release workflow](release-workflow.md) defines the release pull request and publication contract.

## Manage dependencies

Change dependency declarations with uv, then verify the resolved environment:

```bash
uv add package_name
uv remove package_name
uv lock --upgrade-package package_name
uv lock --check
uv pip check
```

Preserve the CPU and GPU source rules in `pyproject.toml`. Treat `uv.lock` as the authority for exact resolved versions.

## Diagnose local runtime failures

Check parser artifacts, GPU policy, and the typed runtime selection without printing credentials:

```bash
uv run python scripts/parser_health.py --check
uv run --no-sync python scripts/test_gpu.py --quick
uv run python -c \
  "from src.config import settings; print(settings.llm_backend, settings.effective_model)"
```

Use the [operations guide](operations-guide.md) for Qdrant, container, snapshot, and backup procedures. Use [GPU setup](gpu-setup.md) for CUDA and Apple Silicon configuration.
