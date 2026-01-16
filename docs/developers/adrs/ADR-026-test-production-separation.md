---
ADR: 026
Title: Test–Production Configuration Separation
Status: Accepted
Version: 1.3
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 024, 014
Tags: testing, config, pydantic, pytest
References:
- [Pydantic — Settings Management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [pytest — Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
---

## Description

Achieve clean test–prod separation by keeping production configuration pure and using pytest fixtures to:

- construct isolated `DocMindSettings` instances when testing pure functions/services, and
- reset the process-global `src.config.settings.settings` singleton **in-place** for tests that exercise UI/runtime code paths that import the singleton.

## Context

Production settings had test‑specific branches causing drift and complexity. We moved test configuration entirely into fixtures.

## Decision Drivers

- No test code in prod
- Simple, standard patterns

## Alternatives

- Test flags in prod settings — rejected
- Separate inheritance tree — overkill

### Decision Framework

| Option                 | Cleanliness (50%) | Simplicity (30%) | Effort (20%) | Total | Decision |
| ---------------------- | ----------------- | ---------------- | ------------ | ----- | -------- |
| Pytest fixtures (Sel.) | 10                | 9                | 7            | 9.3   | ✅ Sel.  |

## Decision

Use pytest fixtures for config; keep prod settings pure.

Important implementation constraint:

- Do **not** rebind `src.config.settings.settings` during tests. Many modules use the recommended import pattern `from src.config import settings`, which captures the singleton instance at import time; rebinding creates stale references and can make the suite order-dependent. Prefer in-place mutation/reset of the existing instance.

## High-Level Architecture

pytest → fixtures → (isolated settings instance OR in-place singleton reset) → tests

## Related Requirements

### Functional Requirements

- FR‑1: Tests use isolated settings instances
- FR‑2: No test code imported in prod modules

### Non-Functional Requirements

- NFR‑1: Simple fixtures; minimal magic

### Performance Requirements

- PR‑1: Fixture setup executes in <50ms typical

### Integration Requirements

- IR‑1: pytest markers `unit|integration|system`; fixtures in `tests/conftest.py`

## Design

### Architecture Overview

- pytest → fixtures configure Pydantic Settings → tests

### Implementation Details

```python
# tests/conftest.py (skeleton)
from pathlib import Path

import pytest
from collections.abc import Iterator

from src.config.settings import DocMindSettings

@pytest.fixture
def app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)  # avoid reading a developer .env file
    monkeypatch.setenv("DOCMIND_LLM_BACKEND", "ollama")
    return DocMindSettings()


@pytest.fixture
def reset_global_settings() -> Iterator[None]:
    import importlib

    def _reset_in_place() -> None:
        settings_mod = importlib.import_module("src.config.settings")
        settings_mod.settings.__init__(_env_file=None)
        settings_mod.reset_bootstrap_state()

    _reset_in_place()
    yield
    _reset_in_place()
```

## Testing

- Ensure no prod files import test helpers
- Verify settings fixture isolation across tests

```python
@pytest.mark.unit
def test_settings_isolation(app_settings):
    assert app_settings.llm_backend == "ollama"


@pytest.mark.integration
def test_ui_with_global_settings(
    reset_global_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that global-singleton settings are reset between tests."""
    import importlib

    monkeypatch.setenv("DOCMIND_LLM_BACKEND", "openai")
    settings_mod = importlib.import_module("src.config.settings")
    settings_mod.bootstrap_settings(force=True)
    assert settings_mod.settings.llm_backend == "openai"
    # reset_global_settings fixture automatically resets on teardown
```

## Consequences

### Positive Outcomes

- Clean prod config; easier maintenance

### Negative Consequences / Trade-offs

- Slight duplication in test fixtures vs prod defaults

### Ongoing Maintenance & Considerations

- Keep fixtures small and explicit; avoid magic env mutation

### Dependencies

- Python: `pydantic>=2`, `pytest>=8`

## Changelog

- 1.3 (2025‑09‑04): Standardized to template; added test fixture example

- 1.2 (2025‑09‑02): Accepted; fixtures isolate settings
