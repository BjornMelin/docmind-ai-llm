---
ADR: 026
Title: Test–Production Configuration Separation
Status: Accepted
Version: 1.2
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 024, 014
Tags: testing, config, pydantic, pytest
References:
- Pydantic Settings
- Pytest fixtures
---

## Description

Achieve clean test–prod separation using pytest fixtures to instantiate isolated Pydantic Settings; remove all test hooks from production config.

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

## High-Level Architecture

pytest → fixtures → settings instance → tests

## Design

### Implementation Details

```python
# tests/conftest.py (skeleton)
@pytest.fixture
def app_settings():
    return make_settings(env={"DOCMIND_DEBUG": "true"})
```

## Testing

- Ensure no prod files import test helpers
- Verify settings fixture isolation across tests

```python
@pytest.mark.unit
def test_settings_isolation(app_settings):
    assert app_settings.debug is True
```

## Consequences

### Positive Outcomes

- Clean prod config; easier maintenance

### Dependencies

- Python: `pydantic>=2`, `pytest>=8`

## Changelog

- 1.2 (2025‑09‑02): Accepted; fixtures isolate settings
