# Configuration Usage Guide

This guide explains how to configure the application using the unified Pydantic Settings (v2) models, environment variables, and the provided `.env.example`.

## Overview

- Configuration is defined in `src/config/settings.py` using nested, typed models.
- A single instance, `settings`, is exported for application code to consume.
- Environment variables map to fields via the prefix `DOCMIND_` and the nested delimiter `__` (double underscore).

Core import:

```python
from src.config.settings import settings
```

## Quick Start

1. Copy the example environment file and edit values as needed:

   ```bash
   cp .env.example .env
   ```

2. Run the app or tests. The settings instance will auto-load from `.env`:

   ```bash
   streamlit run src/app.py
   ```

## Environment Variable Mapping

- Prefix: `DOCMIND_`
- Nested delimiter: `__`

Examples:

- `DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3` → `settings.embedding.model_name`
- `DOCMIND_VLLM__CONTEXT_WINDOW=131072` → `settings.vllm.context_window`
- `DOCMIND_DATABASE__SQLITE_DB_PATH=./data/docmind.db` → `settings.database.sqlite_db_path`
- `DOCMIND_GRAPHRAG_CFG__ENABLED=true` → `settings.graphrag_cfg.enabled`

## Key Sections

- `vllm`: vLLM model and runtime (context window, tokens, temperature, backend).
- `embedding`: BGE‑M3 parameters (model name, dimension, batch sizes).
- `retrieval`: strategy, top_k, reranking controls (mode, normalize, top_n).
- `database`: vector and SQL DB settings (Qdrant, SQLite path).
- `graphrag_cfg`: GraphRAG toggle and graph extraction parameters.
- `ui`: Streamlit options.
- `monitoring`: resource and telemetry limits (memory, VRAM, latency caps).
- `processing`: chunking, overlap, and boundary settings for ingestion.

Refer to `docs/developers/configuration-reference.md` for a complete field list.

## Convenience Aliases

For ergonomic overrides via environment variables the following top‑level aliases are supported; these are applied to the appropriate nested fields during initialization and validated:

- `context_window_size` → `vllm.context_window` (must be > 0)
- `chunk_size` → `processing.chunk_size` (must be > 0)
- `chunk_overlap` → `processing.chunk_overlap` (cannot exceed `chunk_size` when both are provided)
- `enable_multi_agent` → `agents.enable_multi_agent`

Examples:

```env
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_CHUNK_SIZE=1500
DOCMIND_CHUNK_OVERLAP=50
DOCMIND_ENABLE_MULTI_AGENT=true
```

These aliases are optional; prefer nested variables when practical.

## Programmatic Access Patterns

Read nested fields:

```python
from src.config.settings import settings

model_name = settings.embedding.model_name
sqlite_path = settings.database.sqlite_db_path
use_kg = settings.graphrag_cfg.enabled
```

Helper methods for common consumers:

```python
settings.get_model_config()       # {model_name, context_window, max_tokens, temperature, base_url}
settings.get_embedding_config()   # {model_name, device, max_length, batch_size, trust_remote_code}
settings.get_vllm_env_vars()      # env vars for vLLM launchers
```

## .env.example Guidance

- Use `.env.example` as the starting point for local development.
- Copy to `.env` and set values appropriate for your environment.
- Do not commit personal `.env` files.

Recommended minimum overrides for local development:

```env
DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3
DOCMIND_DATABASE__SQLITE_DB_PATH=./data/docmind.db
DOCMIND_RETRIEVAL__TOP_K=10
DOCMIND_GRAPHRAG_CFG__ENABLED=true
```

## Common Overrides

- Increase context window (capped by `settings.llm_context_window_max`):
  - `DOCMIND_VLLM__CONTEXT_WINDOW=131072`
- Disable reranking for performance testing:
  - `DOCMIND_RETRIEVAL__USE_RERANKING=false`
- Toggle GraphRAG (default on per ADR‑019):
  - `DOCMIND_GRAPHRAG_CFG__ENABLED=false`
- Switch multi‑agent mode:
  - `DOCMIND_AGENTS__ENABLE_MULTI_AGENT=false`

## Validation & Errors

- Invalid values raise `ValueError` during settings initialization (e.g., non‑positive chunk sizes, chunk_overlap > chunk_size).
- Directory paths are created if missing for data/cache/logs and SQLite parent directory.

## Best Practices

- Prefer nested environment variables for clarity (`DOCMIND_SECTION__FIELD`).
- Keep `.env` values minimal; default values cover typical local runs.
- Use provided helpers to wire client libraries (e.g., LlamaIndex, vLLM).
- Do not add test‑only flags to configuration; keep tests isolated in code.
