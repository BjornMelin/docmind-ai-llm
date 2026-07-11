# Operations guide

**Version**: 3.0
**Scope**: Local development, restricted networks, and container operations

## Install and run

DocMind uses Python 3.12 or 3.13 and `uv`. The default lockfile path is CPU-first and uses PyTorch's official CPU wheel index.

```bash
uv sync --frozen
cp .env.example .env
uv run streamlit run app.py
```

Use the optional NVIDIA CUDA 12.8 dependency set only on a compatible host:

```bash
uv sync --frozen --no-group cpu --extra gpu
```

The GPU extra is optional. External LLM servers such as vLLM have their own hardware and deployment requirements.

## Prepare models for restricted networks

Fetch the required dense embedding and parser models into their canonical
caches:

```bash
uv run python tools/models/pull.py \
  --bge-m3 \
  --cache_dir ./models_cache \
  --parser-defaults \
  --rapidocr-cache-dir ./cache/models
uv run python scripts/parser_health.py --check
```

`--all` additionally prefetches the pinned SigLIP snapshot used by optional
image retrieval. Text reranking and sparse encoding fail open when their
optional local artifacts are absent.

The parser health command hashes every Docling and RapidOCR file against the canonical source manifest. Mismatches are reported by relative path in `docling.model_issues` and `rapidocr.model_issues`.

After the artifacts are present, set the upstream library offline flags when the deployment requires them:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

These flags control supported Hugging Face clients. They do not prove that the host has no other network path. DocMind separately rejects non-loopback service endpoints unless `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` and the endpoint is allowed by policy.

## Operate the parser

DocMind has one parser path: Docling conversion, pypdfium2 PDF inspection and rasterization, and RapidOCR CPU OCR. There is no parser backend selector or remote parser fallback.

Searchable-PDF export is an optional artifact step:

```bash
uv sync --frozen --extra searchable-pdf
```

OCRmyPDF export is fail-open and runs only on POSIX systems. Windows users can run it through WSL2. Timeout, cancellation, and failure handling terminate and reap the complete subprocess group. A searchable-PDF export failure does not change the parser result.

See [Parser runtime validation](parser-runtime-validation.md) for the model preflight and reproducible benchmark command.

## Operate Qdrant safely

### Start a local instance

Use the repository launcher for local Qdrant:

```bash
./scripts/start_qdrant_local.sh
```

The launcher publishes REST and gRPC exactly on `127.0.0.1:6333` and `127.0.0.1:6334`. If a container with the selected name already exists, the launcher reuses it only when both bindings match exactly. It refuses missing, wildcard, IPv6, remapped, or otherwise different bindings so an operator can inspect and resolve the container without automatic deletion.

The launcher accepts these optional controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_QDRANT_CONTAINER` | `docmind-qdrant-local` | Container name |
| `DOCMIND_QDRANT_IMAGE` | `qdrant/qdrant:v1.10.0` | Container image |
| `DOCMIND_QDRANT_PORT` | `6333` | Loopback REST port |
| `DOCMIND_QDRANT_GRPC_PORT` | `6334` | Loopback gRPC port |
| `DOCMIND_QDRANT_STORAGE` | `./data/qdrant-local` | Host storage path |

### Check collection compatibility

Inspect the configured named-vector collection without mutation:

```bash
uv run python scripts/qdrant_schema.py check
```

Application startup creates a missing collection. It accepts a compatible collection and refuses an incompatible one. Runtime code does not mutate or replace an incompatible schema.

### Rebuild a proven-empty collection

Stop every process that can write to the collection before running:

```bash
uv run python scripts/qdrant_schema.py rebuild-empty
```

`rebuild-empty` reads the exact point count, rebuilds only when that count is zero, and refuses non-empty collections or count failures with exit status 2. Qdrant does not atomically lock the count-and-delete sequence, so writer quiescence is an operator requirement. The command never treats a count error as an empty collection.

Do not delete a collection to resolve a vector mismatch. Preserve or migrate non-empty data, or configure a new collection name.

## Run containers

The production compose overlay runs the application with a read-only root filesystem and writable mounts for required data. The Docker build uses the CPU dependency group unless the build is intentionally configured otherwise.

Container health has one implementation owner:

```bash
python scripts/container_health.py
```

Both the Dockerfile and `docker-compose.prod.yml` invoke this script. It verifies parser model integrity, then opens a TCP connection to the local Streamlit port. It does not make an HTTP request.

## Choose an LLM service

DocMind supports local or policy-approved OpenAI-compatible services. The application does not bundle an inference server.

| Backend | Connection model | Operational note |
| --- | --- | --- |
| Ollama | Native client | Default local backend; DocMind pins the Python client to 0.6.2 |
| vLLM | OpenAI-compatible HTTP | Run and size the server separately |
| llama.cpp | OpenAI-compatible HTTP | Run the server separately |
| LM Studio | OpenAI-compatible HTTP | Run the desktop service separately |

Keep `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` for loopback-only deployments. Enabling a remote service is a deliberate policy change and may also require an endpoint allowlist entry.

## Tune retrieval and agents

SigLIP is the only image embedding backend. ColPali remains an optional visual reranker installed through the `multimodal` extra:

```bash
uv sync --frozen --extra multimodal
```

The agent decision timeout is measured in seconds:

```env
DOCMIND_AGENTS__DECISION_TIMEOUT=200
```

The accepted range is `10s` to `1000s`. Retrieval reranker settings ending in `_MS` remain milliseconds and have separate budgets.

## Protect state

Stop application writers before making a filesystem-level backup. Preserve at least the configured SQLite database, application data, and Qdrant storage. DuckDB-backed ingestion cache data can be included when retaining warm cache state matters.

Production deployments must replace `DOCMIND_HASHING__HMAC_SECRET` with a unique secret of at least 32 bytes. When `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=true`, set `DOCMIND_IMG_AES_KEY_BASE64` to a base64-encoded 32-byte key and manage it outside source control.

## Verify a release candidate

Run the repository's standard quality gates before deployment:

```bash
uv run ruff check .
uv run pyright
uv run python scripts/run_tests.py
uv run python scripts/run_quality_gates.py
```

The wheel smoke script validates wheel contents, metadata, and a `--no-deps` package import. It does not prove that pip can resolve every runtime dependency. Use the locked uv environment or the repository Docker image for supported installations.

See the [configuration reference](configuration.md) for the complete environment schema.
