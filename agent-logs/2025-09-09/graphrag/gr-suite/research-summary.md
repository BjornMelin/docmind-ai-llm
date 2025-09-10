# Research Summary

## Primary Sources

- LlamaIndex docs: PropertyGraphIndex usage, graph stores (SimpleGraphStore), RouterQueryEngine patterns, retriever customization.
- API refs: SimpleGraphStore.get, get_rel_map, persist/from_persist_dir; PropertyGraphIndex.from_documents/from_existing; StorageContext.persist/load_index_from_storage.
- Examples: property_graph_basic/advanced/custom_retriever; RouterQueryEngine; CustomRetrievers (hybrid).
- Blogs: Customizing property graph index (Tomaz Bratanic); additional GraphRAG best practices.

## Key Findings

- Prefer documented store methods (get/properties, get_rel_map) for traversal/exports; avoid internal get_nodes/get_edges.
- Build graph and vector engines as tools for RouterQueryEngine; selectors PydanticSingleSelector (OpenAI) else LLM-based.
- SimpleGraphStore persists to JSON; integrity and atomicity should be added by caller (SnapshotManager pattern).
- Exports: JSONL baseline via get_rel_map; Parquet optional (pyarrow required).
- Hybrid retrievers exist but can be feature-gated until proven value.
- Persistence/versioning and staleness detection are critical for reliability.

Decisions informed by research are captured in final-decisions.md.
