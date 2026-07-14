# Test DocMind changes

This reference maps each test boundary to its supported command. Use the smallest boundary that proves your change, then run the required continuous integration (CI) gates.

## Choose the owning tier

Each tier owns a distinct runtime boundary:

| Tier | Path | Contract |
| --- | --- | --- |
| Unit | `tests/unit` | Domain behavior with controlled dependencies |
| Integration | `tests/integration` | Cross-component wiring and Streamlit AppTest behavior |
| End to end | `tests/e2e` | Application workflows through public surfaces |
| System | `tests/system` | Explicit local-service and runtime behavior |

The end-to-end and system collections are opt-in. Set their collection variables before Pytest starts.

## Run focused tests

Start with the file that owns the changed behavior:

```bash
uv run pytest tests/unit/processing/test_parser_contract.py -vv
```

Run native Pytest lanes when a change crosses several owners:

```bash
uv run pytest tests/unit -q --no-cov
uv run pytest tests/integration -q --no-cov
uv run pytest tests/unit tests/integration -q --no-cov
uv run --no-sync pytest -m requires_gpu --no-cov
```

Run opt-in collections explicitly:

```bash
DOCMIND_RUN_E2E=1 uv run pytest tests/e2e -vv
DOCMIND_RUN_SYSTEM=1 uv run pytest tests/system -vv
```

Required CI jobs establish release readiness. Tests must fail through assertions or uncaught exceptions, not a separate result collector.

## Validate a GPU host

Install the GPU dependency profile before testing real compute hardware:

```bash
uv sync --frozen --no-group cpu --extra gpu
```

Run every test that owns a real GPU boundary:

```bash
uv run --no-sync pytest \
  tests/unit/nlp/test_spacy_service.py \
  tests/integration/core/test_gpu_memory_cleanup_integration.py \
  -m requires_gpu \
  --no-cov \
  -vv
```

Use the hardware wrapper when you also need NVIDIA metadata or video random-access memory (VRAM) sampling:

| Command | Behavior |
| --- | --- |
| `uv run --no-sync python scripts/test_gpu.py --compatibility` | Check NVIDIA and CUDA availability |
| `uv run --no-sync python scripts/test_gpu.py --quick` | Run the focused spaCy CUDA activation test |
| `uv run --no-sync python scripts/test_gpu.py` | Run both GPU test owners |
| `uv run --no-sync python scripts/test_gpu.py --memory-check` | Run both owners, then sample VRAM stability |

`--no-sync` preserves the explicitly installed GPU profile; plain `uv run`
would reconcile the environment back to the default CPU profile. The wrapper
invokes supported Pytest paths through the active Python interpreter. It has no
generic performance mode.

## Mark hardware requirements

Use `requires_gpu` only when a test allocates real GPU resources. Stub device selection in CPU-safe unit tests.

Use the tier markers that match the test path:

- `unit` for isolated domain behavior
- `integration` for cross-component behavior
- `e2e` for application workflows
- `system` for explicit runtime checks

Register any new marker in `pyproject.toml` before using it.

## Keep tests deterministic

Tests should assert observable outcomes at the closest public boundary:

1. Patch network, model, filesystem, clock, and randomness boundaries
2. Use `tmp_path` for files, caches, databases, and persisted state
3. Avoid real sleeps outside explicit hardware or benchmark checks
4. Restore global settings, Streamlit state, and LlamaIndex state through fixtures
5. Assert rendered output, returned values, emitted metadata, or persisted state

Do not leave path fields as `Mock` values. Code can stringify them and create directories named after the mock.

```python
def test_writes_cache(tmp_path):
    settings = Mock()
    settings.cache.dir = tmp_path / "cache"

    write_cache(settings)

    assert settings.cache.dir.is_dir()
```

## Read the canonical testing docs

Use these pages for the full contract:

- `docs/testing/testing-guide.md`: tiers, commands, contribution rules, and release gates
- `docs/testing/test-configuration.md`: profiles, toggles, markers, and hardware setup
- `tests/README.md`: directory ownership map
