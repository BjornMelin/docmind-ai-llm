# Configure DocMind

This reference explains DocMind’s Pydantic Settings model, environment mapping, source precedence, security policy, and high-impact runtime groups. `src/config/settings.py` is the exhaustive field authority.

## Load settings

Application code imports the shared instance:

```python
from src.config import settings

model_name = settings.effective_model
qdrant_url = settings.database.qdrant_url
graphrag_enabled = settings.is_graphrag_enabled()
```

The shared instance does not read `.env` during import. The Streamlit entrypoint calls `bootstrap_settings()` once to load it.

## Understand source precedence

Pydantic Settings resolves values in this order:

1. Explicit constructor arguments
2. Exported `DOCMIND_*` environment variables
3. The selected dotenv file
4. Model defaults

Exported environment variables therefore override `.env` by default.

`DOCMIND_CONFIG__DOTENV_PRIORITY=dotenv_first` changes precedence only for DocMind settings during local development. Security settings remain environment-first.

Use these controls when a third-party library might read a process-global credential:

- `DOCMIND_CONFIG__ENV_MASK_KEYS` removes selected process environment keys
- `DOCMIND_CONFIG__ENV_OVERLAY` copies a validated DocMind setting into a process environment key

Do not use dotenv-first or environment overlays to weaken production security policy.

## Map environment variables

All application settings use the `DOCMIND_` prefix. Use a double underscore for nested fields:

- `DOCMIND_EMBEDDING__MODEL_NAME` maps to `settings.embedding.model_name`
- `DOCMIND_AGENTS__DECISION_TIMEOUT` maps to `settings.agents.decision_timeout`
- `DOCMIND_DATABASE__QDRANT_URL` maps to `settings.database.qdrant_url`

Supported top-level aliases include:

| Alias | Canonical target |
| --- | --- |
| `DOCMIND_MODEL` | `settings.model` (backend-wide override) |
| `DOCMIND_CONTEXT_WINDOW` | `settings.context_window` (global cap override) |
| `DOCMIND_CONTEXT_WINDOW_SIZE` | `settings.context_window` (global cap override) |
| `DOCMIND_CHUNK_SIZE` | `processing.chunk_size` |
| `DOCMIND_CHUNK_OVERLAP` | `processing.chunk_overlap` |
| `DOCMIND_ENABLE_MULTI_AGENT` | `agents.enable_multi_agent` |

Prefer nested names for subsystem-specific settings. Use `DOCMIND_MODEL` and
`DOCMIND_CONTEXT_WINDOW` only when intentionally overriding the selected
backend's defaults.

## Configure core paths

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_APP_NAME` | `DocMind AI` | Application label |
| `DOCMIND_DEBUG` | `false` | Debug logging and diagnostics |
| `DOCMIND_LOG_LEVEL` | `INFO` | Log threshold |
| `DOCMIND_DATA_DIR` | `./data` | Data root |
| `DOCMIND_CACHE__DIR` | `./cache` | Cache root (`settings.cache.dir`) |
| `DOCMIND_LOG_FILE` | `./logs/docmind.log` | Application log |

Bare SQLite filenames move under `DOCMIND_DATA_DIR`. Absolute paths and paths with an explicit parent remain unchanged.

## Configure language model backends

`DOCMIND_LLM_BACKEND` accepts:

- `ollama`
- `vllm`
- `lmstudio`
- `llamacpp`
- `openai_compatible`

The last four use an OpenAI-compatible HTTP boundary. vLLM and llama.cpp run outside the application process.

`settings.effective_model` owns backend-aware model selection. A non-empty
`DOCMIND_MODEL` override wins. Without an override, Ollama uses
`qwen3:4b-instruct`; every other backend uses `DOCMIND_VLLM__MODEL`.

### Configure an OpenAI-compatible endpoint

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_OPENAI__BASE_URL` | `http://localhost:1234/v1` | Provider base URL |
| `DOCMIND_OPENAI__API_KEY` | unset | Bearer credential |
| `DOCMIND_OPENAI__REQUIRE_V1` | `true` | Normalize one trailing `/v1` |
| `DOCMIND_OPENAI__API_MODE` | `chat_completions` | `chat_completions` or `responses` |
| `DOCMIND_OPENAI__DEFAULT_HEADERS` | `{}` | Provider-specific headers |

Local servers may use a non-secret placeholder key. Remote endpoints require an explicit security-policy change.

### Configure vLLM client metadata

| Variable | Default |
| --- | --- |
| `DOCMIND_VLLM__MODEL` | `Qwen/Qwen3-4B-Instruct-2507-FP8` |
| `DOCMIND_VLLM__CONTEXT_WINDOW` | `131072` |
| `DOCMIND_VLLM__MAX_TOKENS` | `2048` |
| `DOCMIND_VLLM__TEMPERATURE` | `0.1` |
| `DOCMIND_VLLM__VLLM_BASE_URL` | `http://localhost:8000` |

FlashInfer, FP8 key-value cache, and GPU memory settings configure the external vLLM process. They do not install or start vLLM inside DocMind.

## Configure endpoint security

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS` | `false` | Permit non-loopback endpoints |
| `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` | Loopback URLs | Allow selected hostnames or URLs |
| `DOCMIND_SECURITY__TRUST_REMOTE_CODE` | `false` | Permit dependency model code |

When remote endpoints are disabled:

- Loopback endpoints are allowed
- A non-loopback hostname must be allowlisted
- DNS resolution must succeed
- Resolved private, link-local, reserved, or otherwise blocked addresses fail validation

Use `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` only when the selected endpoint changes the intended privacy boundary.

## Configure agent deadlines

| Variable | Default | Constraint |
| --- | --- | --- |
| `DOCMIND_AGENTS__ENABLE_MULTI_AGENT` | `true` | Boolean |
| `DOCMIND_AGENTS__DECISION_TIMEOUT` | `200` seconds | 10 to 1,000 seconds |
| `DOCMIND_AGENTS__MAX_RETRIES` | `2` | 0 to 10 |
| `DOCMIND_AGENTS__MAX_CONCURRENT_AGENTS` | `3` | 1 to 10 |
| `DOCMIND_AGENTS__ENABLE_FALLBACK_RAG` | `true` | Boolean |

`decision_timeout` is a supervisor budget in seconds. Provider calls clamp their timeout to the remaining budget.

## Configure document processing

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_PROCESSING__CHUNK_SIZE` | `1500` | TokenTextSplitter chunk size |
| `DOCMIND_PROCESSING__CHUNK_OVERLAP` | `50` | Token overlap |
| `DOCMIND_PROCESSING__NEW_AFTER_N_CHARS` | `1200` | Enrichment boundary |
| `DOCMIND_PROCESSING__COMBINE_TEXT_UNDER_N_CHARS` | `500` | Small-text merge boundary |
| `DOCMIND_PROCESSING__MAX_DOCUMENT_SIZE_MB` | `100` | Source-size limit |
| `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES` | `false` | Encrypt exported page images |

`chunk_overlap` cannot exceed `chunk_size`.

## Configure parsing and OCR

DocMind has one parser contract: Docling, pypdfium2, RapidOCR, and ONNX Runtime.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_PARSING__MAX_PAGES` | `500` | Maximum pages |
| `DOCMIND_PARSING__MAX_RENDER_PIXELS` | `40000000` | Per-render pixel limit |
| `DOCMIND_PARSING__MAX_TOTAL_TEXT_CHARS` | `10000000` | Extracted-text limit |
| `DOCMIND_PARSING__PARSE_TIMEOUT_SECONDS` | `300` | Parser worker deadline |
| `DOCMIND_PARSING__OCRMYPDF_TIMEOUT_SECONDS` | `300` | Searchable-PDF deadline |
| `DOCMIND_PARSING__DIRECT_TEXT_PROBE_BYTES` | `8192` | Binary-content probe |
| `DOCMIND_PDF_BACKEND__RENDER_DPI` | `200` | PDF rasterization resolution |
| `DOCMIND_PDF_BACKEND__MIN_TEXT_CHARS_PER_PAGE` | `24` | Selective OCR threshold |
| `DOCMIND_OCR__FORCE_OCR` | `false` | OCR every PDF page |
| `DOCMIND_OCR__MODEL_CACHE_DIR` | `./cache/models` | Verified model cache |
| `DOCMIND_OCR__SEARCHABLE_PDF_ENABLED` | `false` | Optional OCRmyPDF artifact |
| `DOCMIND_OCR__OCRMYPDF_JOBS` | `1` | OCRmyPDF worker count |

There is no online parser mode or configurable PDF backend. Parser model preflight and integrity checks always apply.

## Configure embeddings and images

Text uses BGE-M3. Sparse retrieval uses direct FastEmbed support. Images use SigLIP.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_EMBEDDING__MODEL_NAME` | `BAAI/bge-m3` | Dense text model |
| `DOCMIND_EMBEDDING__MODEL_REVISION` | `5617a9f…efb181` | Pinned BGE-M3 revision |
| `DOCMIND_EMBEDDING__LOCAL_MODEL_PATH` | unset | Full local SentenceTransformers snapshot |
| `DOCMIND_EMBEDDING__CACHE_FOLDER` | `./models_cache` | Hugging Face model cache |
| `DOCMIND_EMBEDDING__DIMENSION` | `1024` | Text vector dimension |
| `DOCMIND_EMBEDDING__MAX_LENGTH` | `8192` | Native text sequence limit |
| `DOCMIND_EMBEDDING__NORMALIZE_TEXT` | `true` | Normalize dense vectors |
| `DOCMIND_EMBEDDING__ENABLE_SPARSE` | `true` | Sparse retrieval |
| `DOCMIND_EMBEDDING__SIGLIP_MODEL_ID` | `google/siglip-base-patch16-224` | Image model |
| `DOCMIND_EMBEDDING__SIGLIP_MODEL_REVISION` | effective `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` for the default model | Revision override; custom models remain unpinned when unset |
| `DOCMIND_EMBEDDING__BATCH_SIZE_TEXT_CPU` | `4` | CPU text batch |
| `DOCMIND_EMBEDDING__BATCH_SIZE_TEXT_GPU` | `12` | GPU text batch |
| `DOCMIND_EMBEDDING__BATCH_SIZE_IMAGE` | `8` | Image batch |

The curated BGE-M3 model uses its configured revision pin. When the SigLIP
revision is unset and its model ID remains the default, the runtime applies the
repository-owned revision shown above. A custom model ID remains unpinned unless
you also set its revision. Set
`DOCMIND_EMBEDDING__LOCAL_MODEL_PATH` to load a complete local BGE-M3 snapshot
directly; this disables Hub model-ID and revision resolution. Missing local or
cached models leave the embedding runtime unconfigured instead of installing a
mock fallback.

There is no OpenCLIP or alternate image-backend selector.

## Configure retrieval

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_RETRIEVAL__TOP_K` | `10` | Final candidates |
| `DOCMIND_RETRIEVAL__FUSION_MODE` | `rrf` | `rrf` or `dbsf` |
| `DOCMIND_RETRIEVAL__RRF_K` | `60` | RRF constant |
| `DOCMIND_RETRIEVAL__DEDUP_KEY` | `page_id` | `page_id` or `doc_id` |
| `DOCMIND_RETRIEVAL__ENABLE_SERVER_HYBRID` | `false` | Qdrant server fusion |
| `DOCMIND_RETRIEVAL__USE_RERANKING` | `true` | Text reranking |
| `DOCMIND_RETRIEVAL__ENABLE_IMAGE_RETRIEVAL` | `true` | SigLIP image channel |
| `DOCMIND_RETRIEVAL__ENABLE_COLPALI` | `false` | Optional ColPali reranker |
| `DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS` | `250` | Text-stage deadline |
| `DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS` | `150` | SigLIP-stage deadline |
| `DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS` | `800` | Total rerank budget |

Reranking timeouts use milliseconds. Agent `decision_timeout` uses seconds.

## Configure GraphRAG

Both gates default to true:

- `DOCMIND_ENABLE_GRAPHRAG`
- `DOCMIND_GRAPHRAG_CFG__ENABLED`

Graph retrieval requires LlamaIndex core's Property Graph API and a property graph index. Without them, the router uses vector and hybrid tools.

Install the base profile:

```bash
uv sync --frozen
```

See `docs/developers/guides/graphrag.md` for GraphRAG behavior.

## Configure databases and persistence

| Variable | Default |
| --- | --- |
| `DOCMIND_DATABASE__QDRANT_URL` | `http://localhost:6333` |
| `DOCMIND_DATABASE__QDRANT_COLLECTION` | `docmind_docs` |
| `DOCMIND_DATABASE__QDRANT_IMAGE_COLLECTION` | `docmind_images` |
| `DOCMIND_DATABASE__QDRANT_TIMEOUT` | `60` seconds |
| `DOCMIND_DATABASE__SQLITE_DB_PATH` | `docmind.db` |
| `DOCMIND_DATABASE__ENABLE_WAL_MODE` | `true` |

Start the host-local Qdrant service through `scripts/start_qdrant_local.sh`. Docker Compose keeps Qdrant internal to its network.

## Configure an offline run

Install dependencies and model bundles before disconnecting:

```bash
uv sync --frozen
uv run python tools/models/pull.py \
  --bge-m3 \
  --cache_dir ./models_cache \
  --parser-defaults \
  --rapidocr-cache-dir ./cache/models
```

Then set library offline flags:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv run python scripts/parser_health.py --check
```

These controls do not measure network egress. Use a separate network-capture procedure when that evidence is required.

## Validate configuration

Inspect resolved settings without printing secrets:

```python
from src.config import settings

print(settings.llm_backend)
print(settings.backend_base_url_normalized)
print(settings.database.qdrant_url)
print(settings.is_graphrag_enabled())
```

Common validation failures:

| Error | Corrective action |
| --- | --- |
| HMAC secret is shorter than 32 bytes | Supply at least 32 bytes |
| Remote endpoint is rejected | Use loopback or configure the explicit remote policy |
| `chunk_overlap` exceeds `chunk_size` | Lower overlap or raise chunk size |
| Parser models are not ready | Run the parser prefetch command and health check |
