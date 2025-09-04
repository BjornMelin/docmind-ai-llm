---
ADR: 034
Title: Idempotent Indexing & Embedding Reuse
Status: Proposed
Version: 1.1
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 030, 031, 002
Tags: indexing, embeddings, reuse
References:
- [Qdrant — Upsert/Points API](https://qdrant.tech/documentation/concepts/points/)
---

## Description

Avoid re‑embedding unchanged content via stable IDs and content hashes; upsert into Qdrant using stable IDs and update payloads when content changes.

## Context

Re‑embedding unchanged data wastes time and IO; reuse policy is simpler than a separate embedding cache.

## Decision Drivers

- Performance; correctness; simplicity

## Alternatives

- Separate embedding cache — duplicative next to ADR‑030

### Decision Framework

| Option                     | Perf (40%) | Simplicity (30%) | Correctness (20%) | Maintain (10%) | Total | Decision      |
| -------------------------- | ---------- | ---------------- | ----------------- | -------------- | ----- | ------------- |
| Hash+stable IDs (Sel.)     | 9          | 9                | 9                 | 9              | 9.0   | ✅ Selected    |
| Separate embedding cache   | 8          | 5                | 8                 | 5              | 6.8   | Rejected      |

## Decision

Compute and store content hashes and use stable IDs; skip re‑embedding when unchanged; re‑embed on change.

## High-Level Architecture

Ingestion → hash/ID → check store → embed or skip → upsert

## Related Requirements

### Functional Requirements

- FR‑1: Avoid re‑embedding unchanged nodes
- FR‑2: Update payloads on content change

### Non-Functional Requirements

- NFR‑1: Deterministic IDs across runs

### Performance Requirements

- PR‑1: Skip unchanged embeddings; reduce re‑index time ≥50%

### Integration Requirements

- IR‑1: Upsert via Qdrant using stable IDs

## Design

### Architecture Overview

- Ingestion → hash/ID → check store → embed or skip → upsert

### Implementation Details

```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
```

## Consequences

### Positive Outcomes

- Less compute; consistent IDs

### Negative Consequences / Trade-offs

- Requires careful hash policy updates when parsers change

### Ongoing Maintenance & Considerations

- Recompute hashes if tokenization or chunking strategy changes materially

### Dependencies

- Python: `qdrant-client`

## Testing

```python
def test_idempotent_upsert(mocker):
    # stub: ensure upsert called once per unique id
    pass
```

### Configuration

```env
DOCMIND_INDEX__ID_NAMESPACE=docmind
```

## Changelog

- 1.1 (2025‑09‑04): Standardized to template; added decision framework and design stub
- 1.0 (2025‑09‑02): Proposed reuse policy
