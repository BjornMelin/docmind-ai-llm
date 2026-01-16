# User Configuration

This page summarizes common runtime configuration for DocMind AI. For a full developer‑level reference, see [Configuration Guide](../developers/configuration.md). For quick examples, see the Environment Variables section in README.md.

## Core Settings

- LLM provider & base URLs
- Context window and timeouts
- GPU acceleration (CUDA) and memory utilization

## Avoid Global API Keys (Optional)

By default, exported environment variables take precedence over `.env`.

If you have global machine variables like `OPENAI_API_KEY` set but want DocMind
to use only DocMind-scoped configuration, you can opt in to one of these modes:

- Prefer repo `.env` over exported env vars for DocMind settings:
  - `DOCMIND_CONFIG__DOTENV_PRIORITY=dotenv_first`
  - Safety: `security.*` remains env-first.
- Prevent dependencies from using global env vars (allowlist only):
  - `DOCMIND_CONFIG__ENV_MASK_KEYS=OPENAI_API_KEY`
- Provide a compatible env var for dependencies from DocMind settings:
  - Set `DOCMIND_OPENAI__API_KEY=...` and then:
  - `DOCMIND_CONFIG__ENV_OVERLAY=OPENAI_API_KEY:openai.api_key`

These modes are intended for local development; avoid in production.

## Retrieval & GraphRAG

```bash
# Hybrid fusion mode (server-side only)
DOCMIND_RETRIEVAL__FUSION_MODE=rrf   # or dbsf (experimental)

# De-duplication key for fused results
DOCMIND_RETRIEVAL__DEDUP_KEY=page_id  # or doc_id

# Prefetch limits (per-branch)
DOCMIND_RETRIEVAL__PREFETCH_DENSE_LIMIT=200
DOCMIND_RETRIEVAL__PREFETCH_SPARSE_LIMIT=400

# Reranking policy (always-on; override via env)
DOCMIND_RETRIEVAL__USE_RERANKING=true   # set false to disable

# GraphRAG toggle (default ON)
DOCMIND_ENABLE_GRAPHRAG=true          # set false to disable

# Enable server-side hybrid tool (default OFF)
DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=false

# Telemetry (local JSONL; sampling)
DOCMIND_TELEMETRY_SAMPLE=1.0            # 0.0..1.0; 1.0 logs all
DOCMIND_TELEMETRY_DISABLED=false        # true to disable telemetry
DOCMIND_TELEMETRY_ROTATE_BYTES=0        # rotate at N bytes (0 disables)

# Hybrid fusion mode toggle (boolean convenience)
DOCMIND_RETRIEVAL__DBSF_ENABLED=false   # when true, force DBSF
```

Notes:

- Fusion is performed server‑side via the Qdrant Query API; there are no client‑side fusion knobs.
- The knowledge_graph router tool is activated only when a PropertyGraphIndex is present and healthy; traversal depth defaults to path_depth=1.
- When `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID=true`, the router factory registers a
  server-side hybrid tool that executes Qdrant Query API `prefetch` + `Fusion.RRF` (DBSF optional).
  Precedence: an explicit function argument to the router factory always overrides the setting.

## NLP Enrichment (spaCy)

DocMind can optionally enrich ingested text chunks with:

- sentence spans (start/end offsets)
- entity spans (label + text + offsets)

Install a model (recommended for best results):

```bash
uv run python -m spacy download en_core_web_sm
```

Common knobs:

```bash
# Enable/disable enrichment
DOCMIND_SPACY__ENABLED=true

# Device preference: cpu|cuda|apple|auto
DOCMIND_SPACY__DEVICE=auto
DOCMIND_SPACY__GPU_ID=0
```

GPU installs:

- NVIDIA CUDA (Linux/Windows): `uv sync --extra gpu`
- Apple Silicon (macOS arm64): `uv sync --extra apple`

Details: `docs/specs/spec-015-nlp-enrichment-spacy.md` and `docs/developers/gpu-setup.md`.

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
