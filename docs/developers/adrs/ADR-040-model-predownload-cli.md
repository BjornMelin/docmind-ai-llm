---
ADR: 040
Title: Model Pre-download CLI for Offline-First Operation
Status: Accepted
Version: 1.3
Date: 2026-07-14
Supersedes:
Superseded-by:
Related: SPEC-013
Tags: packaging, offline, models
References:
  - https://huggingface.co/docs/huggingface_hub/
---

## Description

Provide a simple CLI to pre-download required model artifacts to support **offline-first** operation and predictable startup behavior.

## Context

DocMind is designed to run fully offline once models are present. Model downloads at runtime cause:

- unpredictable latency and failures
- accidental network usage in offline mode
- inconsistent developer experience

This ADR documents an existing decision already reflected in SPEC-013 and implemented under `tools/models/pull.py`.

## Decision Drivers

- Offline-first defaults (no network required at runtime)
- Deterministic setup and predictable cache locations
- Use maintained libraries (huggingface_hub) rather than bespoke downloaders

## Alternatives

- A: Download lazily at runtime — unpredictable and violates offline-first expectations
- B: Document manual download steps — error-prone
- C: Provide a pre-download CLI (Selected)

### Decision Framework

| Model / Option | Leverage (35%) | Value (25%) | Risk Reduction (25%) | Maint (15%) | Total | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| C: Pre-download CLI | 8 | 8 | 9 | 8 | **8.25** | Selected |
| A: Lazy runtime download | 4 | 6 | 2 | 8 | 4.6 | Rejected |
| B: Manual steps only | 3 | 4 | 4 | 7 | 4.1 | Rejected |

## Decision

Implement `tools/models/pull.py` with complete, pinned snapshot downloads for
the canonical BGE-M3 dense embedding, BM42 sparse model, BGE text reranker, and
SigLIP model. Include verified parser-model prefetch commands and an explicit
`--add` escape hatch for individual files. The runtime resolves the same cache
root and immutable revisions. Do not advertise isolated weight files as
runnable model snapshots.

The repository MUST document offline flags:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## High-Level Architecture

```mermaid
flowchart TD
  A[CLI tools/models/pull.py] --> B[huggingface_hub hf_hub_download]
  B --> C[Local HF cache dir]
  C --> D[Offline runtime (HF_HUB_OFFLINE=1)]
```

## Security & Privacy

- Download sources are public model registries (Hugging Face).
- No document content is involved.
- Runtime should remain offline by default after download.

## Consequences

### Positive Outcomes

- Predictable offline setup
- Reduced runtime failures due to missing weights

### Trade-offs

- Requires users to run a setup step (explicit by design)

## Changelog

- 1.3 (2026-07-14): Added the pinned BM42 source snapshot to the shared model cache.
- 1.2 (2026-07-14): Added the pinned default BGE reranker and one runtime cache owner.
- 1.0 (2026-01-09): Backfilled ADR to match SPEC-013 and existing tooling implementation.
