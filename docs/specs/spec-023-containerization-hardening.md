---
spec: SPEC-023
title: Containerization Hardening — Python 3.11, uv-lock, Non-root, and Canonical DOCMIND Env
version: 1.0.0
date: 2026-01-10
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - NFR-PORT-003: Docker/compose artifacts run and are reproducible from uv.lock.
  - NFR-SEC-001: Default egress disabled; only local endpoints allowed unless explicitly configured.
  - NFR-MAINT-002: Pylint score ≥9.5; Ruff passes.
related_adrs: ["ADR-042", "ADR-024"]
---

## Objective

Make Docker artifacts **runnable, reproducible, and secure-by-default** for DocMind v1:

- Python 3.11 in container (matches repo)
- dependency install via `uv sync --frozen` using `uv.lock`
- Streamlit entrypoint runs `src/app.py`
- non-root runtime
- `.dockerignore` prevents secret leakage and bloated builds
- compose uses `DOCMIND_*` env names

## Non-goals

- Bundling multiple GPU inference backends in compose (high support burden)
- Publishing to a registry (CI publish can be a follow-up ticket)

## Technical design

### Dockerfile

Use a multi-stage Dockerfile:

1. Builder stage:

   - base: Python 3.11 slim (bookworm)
   - install build deps required by Unstructured/PyMuPDF (`libmagic1`, MuPDF libs, etc.)
   - install `uv`
   - set `UV_PYTHON_DOWNLOADS=never` to keep uv from downloading a separate Python
   - prefetch the `torch` wheel with retry/resume before `uv sync` to avoid
     large-download hangs; prefer the PyTorch CPU wheel index when available,
     and fall back to PyPI. Support `TORCH_VERSION`, optional `TORCH_WHEEL_URL`,
     and optional `TORCH_WHEEL_SHA256` for checksum verification.
   - keep the app image CPU-oriented by skipping CUDA-specific `nvidia-*`
     packages during `uv sync` (GPU inference is provided by the Ollama
     service in compose, not the app container)
   - run `uv sync --frozen`

2. Runtime stage:

   - copy venv and application code
   - create non-root user (`docmind`, uid 1000)
   - set `WORKDIR /app`
   - expose port 8501
   - run Streamlit:
     - `streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501`

### docker-compose.yml

Update compose to:

- map `8501:8501`
- optionally include qdrant service (if required)
- use canonical env vars:
  - `DOCMIND_LLM_BACKEND`, `DOCMIND_OLLAMA_BASE_URL`, `DOCMIND_OPENAI__BASE_URL`, etc.
- never hardcode secrets

#### Optional GPU backend profile (Ollama)

Add a compose profile `gpu` that starts a **single** bundled GPU-capable local inference backend:

- `ollama` service (profile: `gpu`)
- GPU access enabled per Docker’s official compose GPU guidance
- Internal-only by default:
  - do not publish `ollama` ports to host unless explicitly desired
  - DocMind connects via the compose network hostname

Recommended env wiring in the `docmind` service:

- `DOCMIND_LLM_BACKEND=ollama`
- `DOCMIND_OLLAMA_BASE_URL=http://ollama:11434`
- keep remote endpoints blocked:
  - `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` (default)
  - extend allowlist to include `ollama`:
    - `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST=["http://localhost","http://127.0.0.1","http://ollama"]`
  - if `DOCMIND_OLLAMA_ENABLE_WEB_SEARCH=true`, add `https://ollama.com` to the
    allowlist and explicitly enable remote endpoints

Power users can still run vLLM/LM Studio externally and point DocMind at them (with allowlist rules), but compose does not bundle them.

Legacy Compose note: older `docker-compose` variants may ignore `deploy.resources`.
If you must support them, use a GPU device request block instead of (or in addition to)
`deploy.resources` in the `ollama` service:

```yaml
device_requests:
  - driver: nvidia
    count: 1
    capabilities: [gpu]
```

#### Hardened production override

Add `docker-compose.prod.yml` for production hardening:

- `read_only: true` for the app container
- `tmpfs` for `/tmp`
- keep `/app/cache` and `/app/logs` as volumes (operators may swap these to `tmpfs` if they prefer ephemeral storage)

This keeps the base compose file dev-friendly while providing a least-privilege option.

### .dockerignore

Add `.dockerignore` to exclude:

- `.git/`, `__pycache__/`, `.venv/`, `htmlcov/`, `data/`, `cache/`, `logs/`
- `.env` and any local secrets
- `opensrc/` (large)

## Security

- Run as non-root user.
- Do not copy `.env` into images by default.
- Avoid `latest` tags for base images; pin to Python 3.11 slim variant.
- Add a healthcheck:
  - Prefer Streamlit’s health endpoint (`/_stcore/health`) when available.
- Keep optional GPU backend internal-only by default (no public ports).

## Testing strategy

Container verification commands:

```bash
docker build -t docmind:dev .
docker compose config
docker compose up --build -d
docker compose logs -f --tail=100 app
```

Hardened compose verification:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml config
```

GPU profile verification (requires Docker Compose v2+ and NVIDIA runtime configured):

```bash
docker compose --profile gpu up --build -d
docker compose ps
docker compose logs -f --tail=100 ollama
```

## RTM updates (docs/specs/traceability.md)

Add a new row:

- NFR-PORT-003: “Dockerfile + compose hardening”
  - Code: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  - Tests: N/A (verification via docker commands)
  - Verification: inspection/manual run
  - Status: Planned → Implemented
