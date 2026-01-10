---
spec: SPEC-030
title: Multimodal Helper Cleanup (Remove Unused `src.utils.multimodal`)
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
related_requirements:
  - NFR-MAINT-003: No TODO/dead code in production modules.
related_adrs: ["ADR-049"]
---

## Objective

Remove `src/utils/multimodal.py` from the production package if it is unused by shipped features, and remove/adjust its tests accordingly.

## Non-goals

- Implementing a new multimodal pipeline (Graph traversal + LLM summarization)
- Changing the existing multimodal reranking architecture (ADR-037 remains)

## Technical design

1. Confirm `src/utils/multimodal.py` is unused in production code:

   - `rg "src\\.utils\\.multimodal" src/` should return nothing (covers direct imports; use `functions.mcp__zen__analyze` if dynamic/string imports are suspected).

2. Delete:

   - `src/utils/multimodal.py`
   - `tests/unit/utils/multimodal/` (if it only tests the deleted module)

3. Update docs referencing the module (handled in WP08 if needed):

   - `docs/developers/system-architecture.md`
   - any prompt/spec templates referencing the module

## Testing strategy

- Run full fast test suite to ensure removal doesn’t break imports.

## RTM updates (docs/specs/traceability.md)

Add a planned row:

- NFR-MAINT-003: “Remove unused multimodal helper”
  - Code: `src/utils/multimodal.py` (deleted)
  - Tests: `tests/unit/utils/multimodal/*` (deleted/updated)
  - Verification: test
  - Status: Planned → Implemented
