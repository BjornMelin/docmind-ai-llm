---
ADR: 029
Title: Boundary‑First Testing Strategy
Status: Accepted
Version: 1.2
Date: 2025-09-04
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

Keep tests simple, fast, and realistic by focusing at system boundaries (API/DB/UI). Prefer real integrations (containers, local services) over mocking internals.

## Decision

- Use pytest as the single test runner.
- Prefer boundary tests; minimize internal mocks.
- Use `responses` for HTTP and `testcontainers` for services as needed.
- Keep suites lean; aim for deterministic, local runs.

## Related Requirements

### Functional Requirements

- FR‑1: Focus tests at boundaries (API/DB/UI).
- FR‑2: Replace internal mocks with real boundaries where practical.

### Non-Functional Requirements

- NFR‑1: Unit <5s; Integration <30s; System <5m.

### Performance Requirements

- PR‑1: Boundary suites fit CI budget (<5m total).

### Integration Requirements

- IR‑1: CI emits junitxml and JSON artifacts; track boundary coverage metrics.
- IR‑2: All suites must meet ADR‑014 gates (quality thresholds, reporting).

## Design

### Core Patterns

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

- Monitor mock counts and effective coverage.

## Consequences

### Positive Outcomes

- Fewer brittle tests; clearer failures

### Negative Consequences / Trade-offs

- Harder to isolate some internal edge cases; cover with unit tests

### Ongoing Maintenance & Considerations

- Periodically audit mock usage; prefer boundary coverage where feasible

### Dependencies

- Python: `pytest>=8`, `responses`, `testcontainers`.

## Changelog

- 1.2 (2025‑09‑04): Consolidated; simplified sections; added IR‑2 linking to ADR‑014.
- 1.1 (2025‑09‑04): Standardized to template; added requirements.
- 1.0 (2025‑08‑29): Accepted boundary strategy.
