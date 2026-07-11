---
spec: SPEC-023
title: Harden the Python 3.12 container and Compose deployment
version: 1.5.0
date: 2026-07-11
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - NFR-PORT-003: Docker/compose artifacts run and are reproducible from uv.lock.
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-MAINT-002: Ruff, strict core Ruff, and Pyright pass.
related_adrs: ["ADR-042", "ADR-024"]
---

## Objective

Make Docker artifacts **runnable, reproducible, and secure-by-default** for DocMind v1:

- Use Python 3.12.13 in the container.
- Install the default CPU dependency group with `uv sync --frozen` and `uv.lock`.
- Run the Streamlit `app.py` entrypoint as a non-root user.
- Exclude local secrets and generated data through `.dockerignore`.
- Use `DOCMIND_*` environment names in Compose.
- Run a query-capable CPU stack by default with one bundled Ollama backend.

## Non-goals

- Bundling multiple language model inference backends in Compose
- Publishing to a registry (CI publish can be a follow-up ticket)

## Technical design

### Dockerfile

Use a multi-stage Dockerfile:

1. Builder stage:

   - base: Python 3.12.13 slim (bookworm)
   - install only the shared libraries required by the locked CPU runtime
     (`libmagic1` and `libgomp1`)
   - copy the pinned official `uv` container binary
   - set `UV_PYTHON_DOWNLOADS=never` to keep uv from downloading a separate Python
   - run `uv sync --frozen`; the default `cpu` dependency group and uv source
     rules select official CPU-only Torch and Torchvision wheels
   - prefetch and verify the Docling and RapidOCR model manifests so local PDF
     parsing is ready in the immutable image; store them under
     `/app/parser-models`, outside the mutable `/app/cache` volume
   - prefetch the pinned BGE-M3 PyTorch snapshot under `/app/hf-models`, exclude
     its duplicate ONNX export, and prove offline construction plus a
     1024-dimensional embedding during the build

2. Runtime stage:

   - copy venv and application code
   - install `libgl1` and `libglib2.0-0`, the native runtimes required by
     RapidOCR's locked `opencv-python` dependency in the slim Debian image;
     this preserves standards-compliant package metadata at the cost of the
     Mesa/X11 layer
   - create non-root user (`docmind`, uid 1000)
   - run `scripts/parser_health.py --check` once in the container entrypoint
     before Streamlit starts
   - set `DOCMIND_OCR__MODEL_CACHE_DIR=/app/parser-models` so a fresh Compose
     cache volume cannot hide the image's verified parser artifacts
   - set `DOCMIND_EMBEDDING__CACHE_FOLDER=/app/hf-models` so the canonical
     embedding adapter loads the immutable BGE-M3 snapshot even when the
     application cache is mounted; retain `LLAMA_INDEX_CACHE_DIR` for
     LlamaIndex-owned artifacts
   - use `scripts/container_health.py` only for the recurring Streamlit TCP
     liveness check
   - set `WORKDIR /app`
   - expose port 8501
   - run Streamlit:
     - `streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.fileWatcherType=none --browser.gatherUsageStats=false`
   - disable Streamlit's source watcher only in the production image; local
     development keeps automatic reloads
   - disable Streamlit usage statistics in the tracked config and container
     command so local-first startup does not emit framework telemetry

CI builds this exact production context with Docker Buildx, loads the result
for the liveness smoke test, and persists reusable layers in the GitHub Actions
cache. The cache avoids downloading and embedding the pinned BGE-M3 snapshot on
every unchanged build without weakening the Dockerfile's offline readiness
assertion.

### Compose configuration

The base Compose file:

- map `8501:8501`
- run CPU Ollama and Qdrant on the internal backend network
- wait for both services to become healthy before starting DocMind
- default `DOCMIND_MODEL` to `qwen3:4b-instruct`
- pin Ollama to `0.31.2` and Qdrant to `v1.18.2`
- use canonical env vars:
  - `DOCMIND_LLM_BACKEND`, `DOCMIND_OLLAMA_BASE_URL`, `DOCMIND_OPENAI__BASE_URL`, etc.
- never hardcode secrets

Start the base stack and pull its model:

```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:4b-instruct
```

#### Optional Ollama GPU override

`docker-compose.gpu.yml` adds the NVIDIA device reservation to the same Ollama service. It does not add another backend or change DocMind's CPU dependency profile.

Start the GPU stack and pull its model:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
docker compose exec ollama ollama pull qwen3:4b-instruct
```

Ollama remains internal-only: Compose does not publish port 11434 to the host. DocMind connects through the Compose network hostname.

Recommended env wiring in the `docmind` service:

- `DOCMIND_LLM_BACKEND=ollama`
- `DOCMIND_OLLAMA_BASE_URL=http://ollama:11434`
- **Important**: `ollama` resolves to a private RFC1918 address inside the compose
  network. When `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false`, DocMind rejects
  hosts that resolve to private/link-local ranges (SSRF/DNS-rebinding hardening),
  even if the hostname is allowlisted. Choose one:
  - **Option A (explicit)**: allow remote endpoints:
    - `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true`
  - **Option B (strict, localhost)**: run DocMind in the same network namespace as
    the `ollama` service and connect via loopback (`DOCMIND_OLLAMA_BASE_URL=http://localhost:11434`).
    This keeps remote endpoints disabled while still allowing the sidecar.
  - If `DOCMIND_OLLAMA_ENABLE_WEB_SEARCH=true`, add `https://ollama.com` to the
    allowlist and enable remote endpoints explicitly.

Power users can run vLLM or LM Studio externally and point DocMind at them. Compose does not bundle those servers.

#### Hardened production override

Add `docker-compose.prod.yml` for production hardening:

- `read_only: true` for the app container
- `tmpfs` for `/tmp`
- keep `/app/cache` and `/app/logs` as volumes (operators may swap these to `tmpfs` if they prefer ephemeral storage)

This keeps the base compose file dev-friendly while providing a least-privilege option.

### Docker build exclusions

Add `.dockerignore` to exclude:

- `.git/`, `__pycache__/`, `.venv/`, `htmlcov/`, `data/`, `cache/`, `logs/`
- `.env` and any local secrets
- `opensrc/` (large)

## Security

- Run as non-root user.
- Do not copy `.env` into images by default.
- Avoid `latest` tags for base images; pin to Python 3.12.13 slim variant.
- Run `python scripts/parser_health.py --check` once before Streamlit starts.
- Run `python scripts/container_health.py` for cheap recurring TCP liveness.
- Keep optional GPU backend internal-only by default (no public ports).
- `scripts/start_qdrant_local.sh` may publish local Qdrant only when both REST
  and gRPC use their configured `127.0.0.1` ports. It refuses existing
  containers with missing, public, additional, or mismatched bindings.

## Testing strategy

Container verification commands:

```bash
docker build -t docmind:dev .
docker compose config
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:4b-instruct
docker compose logs -f --tail=100 app
```

Hardened compose verification:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config
```

GPU override verification requires Docker Compose v2 and an NVIDIA runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
docker compose ps
docker compose logs -f --tail=100 ollama
```

Hosted CI starts Qdrant `v1.18.2` and runs `scripts/qdrant_fusion_smoke.py` with the qdrant-client version resolved from `uv.lock`. The smoke creates a temporary named-vector collection and requires both reciprocal rank fusion (RRF) and distribution-based score fusion (DBSF) queries to return points.

## Traceability update

Add a new row:

- NFR-PORT-003: Dockerfile and Compose hardening
  - Code: `Dockerfile`, `docker-compose.yml`, `docker-compose.gpu.yml`, `.dockerignore`
  - Tests: `tests/unit/scripts/test_container_health.py`, `scripts/qdrant_fusion_smoke.py`
  - Verification: static Compose checks, hosted image liveness, hosted Qdrant fusion smoke
  - Status: Implemented
