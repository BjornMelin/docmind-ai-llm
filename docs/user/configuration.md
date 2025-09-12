# User Configuration

This page summarizes common runtime configuration for DocMind AI. For a full developer‑level reference, see developers/configuration-reference.md. For quick examples, see the Environment Variables section in README.md.

## Core Settings

- LLM provider & base URLs
- Context window and timeouts
- GPU acceleration (CUDA) and memory utilization

## Retrieval & GraphRAG

```bash
# Hybrid fusion mode (server-side only)
DOCMIND_RETRIEVAL__FUSION_MODE=rrf   # or dbsf (experimental)

# De-duplication key for fused results
DOCMIND_RETRIEVAL__DEDUP_KEY=page_id  # or doc_id

# Reranking policy (always-on; override via env)
DOCMIND_RETRIEVAL__USE_RERANKING=true   # set false to disable

# GraphRAG toggle (default ON)
DOCMIND_ENABLE_GRAPHRAG=true          # set false to disable

# Enable server-side hybrid tool (default OFF)
DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=false
```

Notes:

- Fusion is performed server‑side via the Qdrant Query API; there are no client‑side fusion knobs.
- The knowledge_graph router tool is activated only when a PropertyGraphIndex is present and healthy; traversal depth defaults to path_depth=1.
- When `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true`, the router factory registers a
  server-side hybrid tool that executes Qdrant Query API `prefetch` + `Fusion.RRF` (DBSF optional).
  Precedence: an explicit function argument to the router factory always overrides the setting.

## DSPy Optimization (Optional)

```bash
DOCMIND_ENABLE_DSPY_OPTIMIZATION=false
DOCMIND_DSPY_OPTIMIZATION_ITERATIONS=10
DOCMIND_DSPY_OPTIMIZATION_SAMPLES=20
DOCMIND_DSPY_MAX_RETRIES=3
DOCMIND_DSPY_TEMPERATURE=0.1
DOCMIND_DSPY_METRIC_THRESHOLD=0.8
DOCMIND_ENABLE_DSPY_BOOTSTRAPPING=true
```

DSPy runs in the agents layer and augments queries; retrieval (Qdrant + reranking) works independently when DSPy is disabled or unavailable.
