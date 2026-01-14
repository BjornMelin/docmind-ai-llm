# Testing Layout (Unit Tier)

This repository organizes unit tests by domain to mirror the `src/` folder. Subdirectories under `tests/unit/` map directly to source boundaries with a focus on simple, flat groupings where it helps discovery.

- `app/`: App entrypoints and basic wiring (main, app components)
- `agents/`: Coordinator, tools, error recovery, ToolFactory
- `cache/`: Ingestion cache unit tests
- `config/`: Settings and integration‑mapping validation
- `containers/`: Container wiring tests split out from `core/`
- `core/`: Core invariants, exception handling, dependencies, spaCy manager
- `integrations/`: Cross‑cutting adapters (e.g., DSPy retriever)
- `models/`: Pydantic models (schemas, processing, storage)
  - `models/embeddings/`: Text/image/unified embedder suites
- `processing/`: Document processing; Unstructured + LlamaIndex pipeline
- `prompts/`: Prompt templates
- `retrieval/`: Retrieval domain
  - `query_engine/`: Query engine behaviors and fallbacks
  - `qdrant/`: Qdrant prefetch/dedup behavior
  - `dedup/`: Dedup helpers prior to final cut
  - `embeddings/`: Retrieval‑side embedding contracts
  - `pipeline/`: Server hybrid pipeline assembly
  - `sparse/`: FastEmbed/BM25 sparse flow
  - `telemetry/`: Retrieval telemetry assertions
  - `rbac/`: Owner filters and RBAC
  - `reranking/`: Text/visual reranking
    - `text/`, `visual/`, `siglip/`, `rrf/`, `infra/`
- `telemetry/`: Global telemetry toggles and schema contracts
- `ui/`: UI utility/component tests
- `utils/`: Utilities split by feature
  - `core/`, `document/`, `monitoring/`, `multimodal/`, `security/`, `siglip_adapter/`, `storage/`

## Fixture strategy

- `tests/unit/conftest.py`: minimal unit-wide defaults only.
- Domain `conftest.py` files live beside their tests (e.g., `processing/conftest.py`, `utils/conftest.py`) and are not imported across domains.
- Avoid global fixture sprawl; keep scopes tight and deterministic (no real network/sleeps/GPU).

## Running tests

- Unit with coverage: `uv run pytest -q tests/unit --cov=src` (note: `-q` for clean CI; use `-v` for verbose debugging)
- Integration (offline) lives in `tests/integration` (separate job in CI).

## Pytest settings

- Import mode: `--import-mode=importlib` to avoid path quirks.
- Test discovery roots: `[tool.pytest.ini_options].testpaths = ["tests"]`.
