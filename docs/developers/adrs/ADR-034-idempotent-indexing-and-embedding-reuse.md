# ADR-034: Idempotent Indexing & Embedding Reuse

## Metadata

**Status:** Proposed  
**Version/Date:** v1.0 / 2025-09-02

## Title

Stable Hashing and Upsert Policy to Avoid Re-Embedding Unchanged Content

## Description

Standardize idempotent indexing so that unchanged documents and nodes are not re-embedded or re-indexed. Use content hashing and stable node IDs to detect unchanged content and perform Qdrant upserts keyed by stable IDs. No separate "embedding cache" is introduced; this policy works alongside ADR-030’s IngestionCache for processing intermediates.

## Context

- Archived ADR-007 implied embedding reuse.  
- ADR-030 unified the processing cache; ADR-031 clarified persistence boundaries.  
- We need a simple, robust policy for avoiding unnecessary re-embedding to reduce compute and IO.

## Decision Drivers

- Performance and cost reduction on re-runs.
- Correctness: do not miss updates when content changes.
- Simplicity: leverage LlamaIndex node metadata and Qdrant upsert semantics.

## Alternatives

- Dedicated embedding cache (rejected: duplicative alongside ADR-030; adds complexity).  
- Ad-hoc skip logic (rejected: inconsistent and error-prone).

## Decision

- Compute and persist a content hash (SHA-256) for each document and node.  
- Use stable node IDs (e.g., based on file path + chunk offset) or LlamaIndex-provided node IDs consistently.  
- Store the content hash in node metadata and in Qdrant payloads.  
- On re-indexing, if hash unchanged, skip re-embedding and upsert unless forced.  
- On hash change, re-embed and upsert with the same stable ID, updating payloads.

## Design

### Hashing

- **Document-level**: SHA-256 of file bytes.  
- **Node-level**: SHA-256 of node text and salient metadata.  
- Fallback heuristic when bytes unavailable: mtime + size + path.

### IDs

- Stable ID derived from `(normalized_path, chunk_index)` when not provided by LlamaIndex.  
- Ensure deterministic ID generation across runs.

### Upserts

- Qdrant upsert keyed by the stable ID.  
- If record exists and hash unchanged → skip heavy work.  
- If record exists and hash changed → re-embed and update payloads.

### Forcing Rebuilds

- Provide a force flag to recompute everything when required (e.g., parameter changes).

## Testing

- **Unit**: unchanged document re-run performs no embedding; changed document re-run performs embedding.  
- **Integration**: index built → re-run no-op; edit file → re-run updates only changed nodes.

## Dependencies

- None beyond existing LlamaIndex + Qdrant stack.

## Related Decisions

- **ADR-009** (Document Processing): Produces nodes and metadata.  
- **ADR-002** (Unified Embeddings): Embedding model inputs/outputs.  
- **ADR-031** (Persistence Architecture): Clarifies vector store and cache separation.  
- **ADR-030** (Cache Unification): Coexists with processing cache without duplication.

## Consequences

- **Positive**: Faster re-runs; reduced compute; simpler operational story.  
- **Trade-offs**: Requires careful ID/hash consistency; need force-rebuild path for param changes.

## Changelog

- **v1.0 (2025-09-02)**: Initial proposal for idempotent indexing and reuse policy.
