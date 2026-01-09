---
spec: SPEC-023
title: Containerization Hardening — Python 3.11, uv-lock, Non-root, and Canonical DOCMIND Env
version: 1.0.0
date: 2026-01-09
owners: ["ai-arch"]
status: Draft
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

- Supporting every GPU configuration in Docker for v1 (GPU compose profiles can be added later)
- Publishing to a registry (CI publish can be a follow-up ticket)

## Technical design

### Dockerfile

Use a multi-stage Dockerfile:

1. Builder stage:

   - base: Python 3.11 slim (bookworm)
   - install build deps required by Unstructured/PyMuPDF (`libmagic1`, MuPDF libs, etc.)
   - install `uv`
   - run `uv sync --frozen` (optionally with extras for GPU in a separate variant)

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

## Testing strategy

Container verification commands:

```bash
docker build -t docmind:dev .
docker compose config
docker compose up --build -d
docker compose logs -f --tail=100 app
```

## RTM updates (docs/specs/traceability.md)

Add a new row:

- NFR-PORT-003: “Dockerfile + compose hardening”
  - Code: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  - Tests: N/A (verification via docker commands)
  - Verification: inspection/manual run
  - Status: Planned → Implemented
