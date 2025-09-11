 (canonical env; no UI toggle)# Configuration Usage Guide
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)This guide explains how to configure the application using the unified Pydantic Settings (v2) models, environment variables, and the provided `.env.example`.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Overview
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Configuration is defined in `src/config/settings.py` using nested, typed models.
 (canonical env; no UI toggle)- A single instance, `settings`, is exported for application code to consume.
 (canonical env; no UI toggle)- Environment variables map to fields via the prefix `DOCMIND_` and the nested delimiter `__` (double underscore).
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Core import:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)```python
 (canonical env; no UI toggle)from src.config.settings import settings
 (canonical env; no UI toggle)```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Quick Start
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)1. Copy the example environment file and edit values as needed:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)   ```bash
 (canonical env; no UI toggle)   cp .env.example .env
 (canonical env; no UI toggle)   ```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)2. Run the app or tests. The settings instance will auto-load from `.env`:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)   ```bash
 (canonical env; no UI toggle)   streamlit run src/app.py
 (canonical env; no UI toggle)   ```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Environment Variable Mapping
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Prefix: `DOCMIND_`
 (canonical env; no UI toggle)- Nested delimiter: `__`
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Examples:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- `DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3` → `settings.embedding.model_name`
 (canonical env; no UI toggle)- `DOCMIND_VLLM__CONTEXT_WINDOW=131072` → `settings.vllm.context_window`
 (canonical env; no UI toggle)- `DOCMIND_DATABASE__SQLITE_DB_PATH=./data/docmind.db` → `settings.database.sqlite_db_path`
 (canonical env; no UI toggle)- `DOCMIND_GRAPHRAG_CFG__ENABLED=true` → `settings.graphrag_cfg.enabled`
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Key Sections
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- `vllm`: vLLM model and runtime (context window, tokens, temperature, backend).
 (canonical env; no UI toggle)- `embedding`: BGE‑M3 parameters (model name, dimension, batch sizes).
 (canonical env; no UI toggle)- `retrieval`: strategy, top_k, reranking controls (mode, normalize, top_n).
 (canonical env; no UI toggle)- `database`: vector and SQL DB settings (Qdrant, SQLite path).
 (canonical env; no UI toggle)- `graphrag_cfg`: GraphRAG toggle and graph extraction parameters.
 (canonical env; no UI toggle)- `ui`: Streamlit options.
 (canonical env; no UI toggle)- `monitoring`: resource and telemetry limits (memory, VRAM, latency caps).
 (canonical env; no UI toggle)- `processing`: chunking, overlap, and boundary settings for ingestion.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Refer to `docs/developers/configuration-reference.md` for a complete field list.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Convenience Aliases
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)For ergonomic overrides via environment variables the following top‑level aliases are supported; these are applied to the appropriate nested fields during initialization and validated:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- `context_window_size` → `vllm.context_window` (must be > 0)
 (canonical env; no UI toggle)- `chunk_size` → `processing.chunk_size` (must be > 0)
 (canonical env; no UI toggle)- `chunk_overlap` → `processing.chunk_overlap` (cannot exceed `chunk_size` when both are provided)
 (canonical env; no UI toggle)- `enable_multi_agent` → `agents.enable_multi_agent`
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Examples:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)```env
 (canonical env; no UI toggle)DOCMIND_CONTEXT_WINDOW_SIZE=65536
 (canonical env; no UI toggle)DOCMIND_CHUNK_SIZE=1500
 (canonical env; no UI toggle)DOCMIND_CHUNK_OVERLAP=50
 (canonical env; no UI toggle)DOCMIND_ENABLE_MULTI_AGENT=true
 (canonical env; no UI toggle)```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)These aliases are optional; prefer nested variables when practical.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Programmatic Access Patterns
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Read nested fields:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)```python
 (canonical env; no UI toggle)from src.config.settings import settings
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)model_name = settings.embedding.model_name
 (canonical env; no UI toggle)sqlite_path = settings.database.sqlite_db_path
 (canonical env; no UI toggle)use_kg = settings.graphrag_cfg.enabled
 (canonical env; no UI toggle)```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Helper methods for common consumers:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)```python
 (canonical env; no UI toggle)settings.get_model_config()       # {model_name, context_window, max_tokens, temperature, base_url}
 (canonical env; no UI toggle)settings.get_embedding_config()   # {model_name, device, max_length, batch_size, trust_remote_code}
 (canonical env; no UI toggle)settings.get_vllm_env_vars()      # env vars for vLLM launchers
 (canonical env; no UI toggle)```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## .env.example Guidance
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Use `.env.example` as the starting point for local development.
 (canonical env; no UI toggle)- Copy to `.env` and set values appropriate for your environment.
 (canonical env; no UI toggle)- Do not commit personal `.env` files.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)Recommended minimum overrides for local development:
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)```env
 (canonical env; no UI toggle)DOCMIND_EMBEDDING__MODEL_NAME=BAAI/bge-m3
 (canonical env; no UI toggle)DOCMIND_DATABASE__SQLITE_DB_PATH=./data/docmind.db
 (canonical env; no UI toggle)DOCMIND_RETRIEVAL__TOP_K=10
 (canonical env; no UI toggle)DOCMIND_GRAPHRAG_CFG__ENABLED=true
 (canonical env; no UI toggle)```
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Common Overrides
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Increase context window (capped by `settings.llm_context_window_max`):
 (canonical env; no UI toggle)  - `DOCMIND_VLLM__CONTEXT_WINDOW=131072`
 (canonical env; no UI toggle)- Disable reranking for performance testing:
 (canonical env; no UI toggle)  - `DOCMIND_RETRIEVAL__USE_RERANKING=false`
 (canonical env; no UI toggle)- Toggle GraphRAG (default on per ADR‑019):
 (canonical env; no UI toggle)  - `DOCMIND_GRAPHRAG_CFG__ENABLED=false`
 (canonical env; no UI toggle)- Switch multi‑agent mode:
 (canonical env; no UI toggle)  - `DOCMIND_AGENTS__ENABLE_MULTI_AGENT=false`
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Validation & Errors
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Invalid values raise `ValueError` during settings initialization (e.g., non‑positive chunk sizes, chunk_overlap > chunk_size).
 (canonical env; no UI toggle)- Directory paths are created if missing for data/cache/logs and SQLite parent directory.
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)## Best Practices
 (canonical env; no UI toggle)
 (canonical env; no UI toggle)- Prefer nested environment variables for clarity (`DOCMIND_SECTION__FIELD`).
 (canonical env; no UI toggle)- Keep `.env` values minimal; default values cover typical local runs.
 (canonical env; no UI toggle)- Use provided helpers to wire client libraries (e.g., LlamaIndex, vLLM).
 (canonical env; no UI toggle)- Do not add test‑only flags to configuration; keep tests isolated in code.
