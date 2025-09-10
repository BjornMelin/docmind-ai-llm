# GraphRAG Finalization — Cross-Reference and Spec/ADR/Req Updates

Date: 2025-09-09
Owner: Eng/Arch

## Scope

- Cross-reference `docs/specs/spec-006-graphrag.md`, `docs/specs/requirements.md`, ADRs (notably `ADR-019` and `ADR-024`), and our research plan in `agent-logs/2025-09-09/graphrag/001-audit-and-plan.md`.
- Ensure all documents are aligned with a library-first, portable GraphRAG approach using documented LlamaIndex APIs only.
- Record the exact changes made to specs/ADRs/requirements.

## Summary of Changes

- SPEC-006 updated:
  - Removed dependency on a custom synonym retriever and extra `graph_retriever.py` wrapper.
  - Mandated use of `PropertyGraphIndex.as_retriever(...)` for graph-aware retrieval.
  - Standardized exports to JSONL/Parquet derived from `property_graph_store.get_rel_map(...)`.
  - Added explicit References to LlamaIndex docs and examples.

- Requirements updated (FR-009):
  - Clarified that GraphRAG is optional, uses only documented LlamaIndex APIs (`as_retriever`, `get_rel_map`), and shall export JSONL/Parquet (PyArrow optional) from `get_rel_map`.

- New references and additions:
  - Added ADR‑038 (GraphRAG router + persistence) and SPEC‑014 (SnapshotManager) to align UI toggle, router fallback, staleness badge, and export formats across docs.

- ADR-019 updated to v3.2:
  - Codified a library-first API policy (no index mutation; use documented APIs only).
  - Specified portable exports via `get_rel_map` (JSONL/Parquet) and optional `save_networkx_graph` HTML for inspection.
  - Noted helpers should be wrappers/pure functions; legacy attachment is test-only.

- ADR-024 remains consistent:
  - `DOCMIND_GRAPHRAG__ENABLED` config switch is documented. No changes required.

## Rationale

- Using undocumented store internals (`get_nodes`, `get_edges`) risks incompatibilities across graph stores (Neo4j, Nebula, FalkorDB, TiDB) and versions.
- Mutating `PropertyGraphIndex` instances is brittle; wrappers keep code clean and testable.
- Exports via `get_rel_map` ensure portability and stable schemas across backends.

## References

- LlamaIndex Property Graph Guide: <https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide>
- Property Graph Examples: <https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_basic>
- Graph store API (`get`, `get_rel_map`, `save_networkx_graph`): Context7 code snippets referenced in research.
- GraphRAG examples (optional advanced): GraphRAG v2, Agentic GraphRAG Vertex (Context7 examples).

## Acceptance Criteria for Document Alignment

- [x] SPEC-006 reflects library-first retriever and exports via `get_rel_map`.
- [x] FR-009 updated to remove synonym retriever and mandate portable exports.
- [x] ADR-019 explicitly prohibits index mutation and undocumented store APIs; adds export policy.
- [x] `DOCMIND_GRAPHRAG__ENABLED` flag remains authoritative per ADR-024.
- [x] Research plan in `agent-logs/2025-09-09/graphrag/001-audit-and-plan.md` is fully incorporated.

## Next Steps (Implementation)

- Implement Phase 1 changes per `001-audit-and-plan.md` (helpers, exports, async offloading, removal of internals) with updated tests.
- Stage Phase 2 (optional GraphRAG) as a modular flag-gated addition after Phase 1 stabilizes.
