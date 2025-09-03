# Testing Layout (Unit Tier)

This repository organizes unit tests by domain to mirror the src/ folder. This keeps ownership clear and reduces fixture sprawl. Subdirectories under tests/unit/ map directly to source boundaries.

- `app/`: App entrypoints and basic wiring (main, app components)
- `agents/`: Coordinator, tools, error recovery, ToolFactory
- `cache/`: Simple cache and cache interfaces
- `config/`: Settings and validation
- `core/`: Containers, exceptions, invariants; `core/infrastructure/`: spaCy manager, resources
- `embeddings/`: Cross-boundary embedding contracts (dims/shape)
- `models/`: Pydantic models (schemas, processing, storage, embeddings)
- `processing/`: Document processing; processing/chunking and processing/embeddings
- `prompts/`: Prompt templates
- `retrieval/`: Query engine, hybrid search; retrieval/reranking, retrieval/similarity, retrieval/vector_store, retrieval/adapters (e.g., DSPy)
- `storage/`: Persistence and managers
- `utils/`: Monitoring, multimodal and core utility tests (with local fixtures)

## Fixture strategy

- `tests/unit/conftest.py`: minimal unit-wide defaults only.
- Domain `conftest.py` files live beside their tests (e.g., `processing/conftest.py`, `utils/conftest.py`) and are not imported across domains.
- Avoid global fixture sprawl; keep scopes tight and deterministic (no real network/sleeps/GPU).

## Running tests

- Unit with coverage: `uv run pytest tests/unit -m unit --cov=src -q`
- Integration (offline) lives in `tests/integration` (separate job in CI).

## Pytest settings

- Import mode: `--import-mode=importlib` to avoid path quirks.
- Test discovery roots: `[tool.pytest.ini_options].testpaths = ["tests"]`.
