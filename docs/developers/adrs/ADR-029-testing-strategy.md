---
ADR: 029
Title: Modern Testing Strategy with Boundary Testing
Status: Accepted
Version: 1.1
Date: 2025-08-29
Supersedes:
Superseded-by:
Related: 014, 026
Tags: testing, boundary, pytest
References:
- [pytest — Official Docs](https://docs.pytest.org/)
- [responses — Requests Mocking](https://github.com/getsentry/responses)
- [testcontainers — Python](https://testcontainers-python.readthedocs.io/)
---

## Description

Adopt boundary testing patterns; reduce internal mocking; set realistic coverage goals. Keep tests fast and focused on boundaries.

## Context

Over‑mocked tests were brittle and noisy. Boundary tests cut mocks and raise real coverage.

## Decision Drivers

- Maintainability; reliability; dev speed

## Alternatives

- Keep heavy mocks — brittle
- Rewrite from scratch — risky

### Decision Framework

| Option              | Maintain (35%) | Reliability (25%) | Effort (20%) | Coverage (20%) | Total | Decision |
| ------------------- | ------------- | ----------------- | ------------ | -------------- | ----- | -------- |
| Boundary + libs     | 9             | 9                 | 7            | 8              | 8.3   | ✅ Sel.  |

## Decision

Use pytest + responses/testcontainers where needed; avoid mocking internals.

## High-Level Architecture

tests → boundaries (API/DB/UI) → metrics

## Related Requirements

### Functional Requirements

- FR‑1: Focus tests at boundaries (API/DB/UI)
- FR‑2: Replace internal mocks with real boundaries

### Non-Functional Requirements

- NFR‑1: Unit <5s; Integration <30s; System <5m

### Integration Requirements

- IR‑1: pytest markers and CI gates via junitxml

## Design

### Implementation Details

```python
@pytest.mark.integration
def test_vector_store_boundary(qdrant):
    # assert retrieval under real client
    pass

@pytest.mark.integration
def test_qdrant_with_container(qdrant_container):
    from qdrant_client import QdrantClient
    c = QdrantClient(url=f"http://localhost:{qdrant_container.port}")
    assert c.get_collections()

def test_streamlit_page_boot(app_runner):
    resp = app_runner.open("/chat")
    assert resp.status_code == 200
```

## Testing

- Monitor mock counts and effective coverage

## Consequences

### Positive Outcomes

- Fewer brittle tests; clearer failures

### Dependencies

- Python: `pytest>=8`, `responses`, `testcontainers`

## Changelog

- 1.1 (2025‑09‑04): Standardized to template; added requirements

- 1.0 (2025‑08‑29): Accepted boundary strategy
