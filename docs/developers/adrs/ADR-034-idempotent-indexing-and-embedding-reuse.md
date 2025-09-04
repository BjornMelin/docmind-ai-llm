---
ADR: 034
Title: Idempotent Indexing & Embedding Reuse
Status: Proposed
Version: 1.0
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 002, 009, 030, 031
Tags: indexing, idempotency, embeddings, qdrant
References:
- [hashlib — SHA-256](https://docs.python.org/3/library/hashlib.html)
- [Qdrant — Upsert](https://qdrant.tech/documentation/concepts/points/#upsert)
- [LlamaIndex — Nodes & Metadata](https://docs.llamaindex.ai/)
---

## Description

Avoid re-embedding and re-indexing unchanged content by using stable content hashing, deterministic node IDs, and Qdrant upserts keyed by stable IDs. No separate "embedding cache" is introduced; this policy complements ADR‑030’s processing cache.

## Context

Re-indexing unchanged content wastes compute and I/O. Prior ADRs unified processing cache (ADR‑030) and clarified storage (ADR‑031). We need a simple, robust reuse policy that prevents redundant embeddings while guaranteeing updates when content changes.

## Decision Drivers

- Reduce re-run cost (time/compute)
- Ensure correctness on content change
- KISS: leverage existing libraries and metadata

## Alternatives

- A: Dedicated embedding cache — Pros: explicit; Cons: duplicates ADR‑030; added complexity
- B: Ad-hoc skip logic — Pros: easy start; Cons: inconsistent, error-prone
- C: Stable hash + upsert (Selected) — Pros: simple, robust, library-first

### Decision Framework

| Model / Option                 | Simplicity (35%) | Correctness (35%) | Performance (20%) | Maintenance (10%) | Total Score | Decision      |
| ------------------------------ | ---------------- | ----------------- | ----------------- | ----------------- | ----------- | ------------- |
| Stable hash + upsert (Sel.)    | 9                | 9                 | 9                 | 9                 | **9.0**     | ✅ Selected    |
| Dedicated embedding cache      | 6                | 8                 | 8                 | 5                 | 6.9         | Rejected      |
| Ad-hoc skip logic              | 7                | 5                 | 6                 | 6                 | 6.1         | Rejected      |

## Decision

Compute content hashes (document- and node-level) and use deterministic node IDs. Store hashes in node metadata and Qdrant payloads. On re-index, skip embedding and upsert when hashes are unchanged; otherwise, re-embed and upsert using the same stable ID.

## High-Level Architecture

```mermaid
graph TD
  A[File] --> B[Process (ADR‑009)]
  B --> C[Hash + Stable IDs]
  C -->|changed| D[Embed + Upsert]
  C -->|unchanged| E[Skip Embed; Upsert/No-op]
  D --> Q[Qdrant]
  E --> Q
```

## Related Requirements

### Functional Requirements

- FR‑1: Compute and persist document and node content hashes
- FR‑2: Generate deterministic node IDs across runs
- FR‑3: Upsert to Qdrant keyed by stable ID; update when changed

### Non-Functional Requirements

- NFR‑1: No separate embedding cache layer
- NFR‑2: Minimal code; rely on library features

### Performance Requirements

- PR‑1: Re-run unchanged corpus performs near no-op indexing

### Integration Requirements

- IR‑1: Use LlamaIndex node metadata; Qdrant payload for hash

## Design

### Architecture Overview

- Hash at document (bytes) and node (text + salient metadata) levels
- Deterministic node IDs (e.g., normalized_path + chunk_index) if not provided
- Qdrant upsert keyed by ID; compare stored hash to decide work

### Implementation Details

In `src/core/indexing.py` (illustrative):

```python
import hashlib

def node_hash(text: str, meta: dict) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(meta.get("section","?").encode("utf-8"))
    return h.hexdigest()
```

### Configuration

```env
DOCMIND_INDEX__FORCE_REINDEX=false
```

## Testing

```python
def test_reuse_policy(indexer, tmp_docs):
    # initial index -> re-run is no-op; modify one file -> only changed nodes update
    pass
```

## Consequences

### Positive Outcomes

- Faster re-runs; reduced compute and I/O
- Simple to reason about; minimal moving parts

### Negative Consequences / Trade-offs

- Requires consistent ID/hash behavior across versions
- Force rebuilds needed when parameters change

### Ongoing Maintenance & Considerations

- Keep hash inputs stable; version if schema changes
- Provide explicit force-reindex path

### Dependencies

- Python: `hashlib`; existing LlamaIndex + Qdrant stack

## Changelog

- **v1.0 (2025-09-02)**: Initial proposal for idempotent indexing and reuse policy.
