---
ADR: 042
Title: Containerization hardening with CPU and GPU Ollama modes
Status: Implemented
Version: 1.6
Date: 2026-07-11
Supersedes: 015
Superseded-by:
Related: 015, 024
Tags: docker, compose, packaging, security, deployment
References:
  - https://docs.docker.com/build/building/multi-stage/
  - https://docs.docker.com/compose/
  - https://docs.docker.com/compose/how-tos/gpu-support/
  - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
  - https://docs.astral.sh/uv/guides/integration/docker/
---

## Description

The implemented container baseline uses a **Python 3.12.13** `uv`-based,
multi-stage Dockerfile and a Compose configuration that:

- uses canonical `DOCMIND_*` env vars
- is secure-by-default (non-root, `.dockerignore`, no baked secrets)
- runs one bundled Ollama service on CPU by default
- adds NVIDIA access to the same service through `docker-compose.gpu.yml`
- keeps Ollama and Qdrant on the internal network

## Context

Before this decision, the shipped `Dockerfile` and `docker-compose.yml` had
drifted from repository constraints:

- Docker CMD is invalid (shell string inside JSON form)
- entrypoint path does not match Streamlit entry (`app.py`)
- `docker-compose.yml` uses non-canonical env variable names (not `DOCMIND_*`)
- `.dockerignore` is missing, increasing secret leak risk

The container contract must run a query-capable CPU stack without forcing CUDA or PyTorch GPU packages into the application image. An explicit Compose override adds NVIDIA access to the same Ollama service. Operators configure vLLM and other OpenAI-compatible servers outside the DocMind container.

## Decision Drivers

- Correctness: containers must run out-of-the-box
- Reproducibility: deterministic dependency install via `uv.lock`
- Security: non-root runtime, .dockerignore, avoid leaking `.env`
- Maintainability: keep Docker artifacts minimal and aligned with repo config discipline
- GPU capability: provide a supported local GPU path without coupling the app image to CUDA/toolkit churn

Configure vLLM model, context, cache, and hardware settings on the external server. Keep Ollama configuration focused on its service endpoint through `DOCMIND_OLLAMA_BASE_URL`.

## Alternatives

- A: Minimal patch existing Dockerfile/compose (still leaves security/size debt; unclear GPU story)
- B: Single Dockerfile + compose with multiple GPU backends bundled (Ollama + vLLM) (more capability, but high support burden)
- C: GPU-enabled DocMind app image (CUDA/PyTorch inside) (brittle; huge; couples releases to GPU stack)
- D: App container plus one CPU Ollama service and a minimal GPU override (Selected)

### Decision Framework (≥9.0)

Weights: Complexity 40% · Perf/Size 30% · Alignment/Security 30% (10 = best)

| Option | Complexity (40%) | Perf/Size (30%) | Alignment/Security (30%) | Total | Decision |
| --- | --- | --- | --- | --- | --- |
| D: App container + CPU Ollama + GPU override | 9.5 | 8.5 | 9.5 | **9.2** | Selected |
| B: App container + multiple GPU backends bundled | 6.5 | 7.5 | 7.5 | 7.1 | Rejected |
| C: GPU-enabled app image | 5.5 | 6.0 | 7.0 | 6.1 | Rejected |
| A: Minimal patch only | 8.0 | 5.0 | 6.0 | 6.7 | Rejected |

## Decision

Implement a ship-ready container baseline:

- Add `.dockerignore`
- Replace Dockerfile with:
  - Python 3.12.13 base
  - multi-stage build with `uv sync --frozen`
  - `UV_PYTHON_DOWNLOADS=never` to force uv to use container Python
  - non-root runtime user
  - runtime `libgl1` and `libglib2.0-0` for the locked RapidOCR/OpenCV import
    contract on Debian slim; a uv exclusion plus the headless wheel was
    rejected because it makes `uv pip check` report RapidOCR's installed
    metadata as unsatisfied
  - correct Streamlit entrypoint: `streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.fileWatcherType=none --browser.gatherUsageStats=false`
  - disable source watching in the immutable production image to avoid
    unnecessary module introspection; keep local development reloads enabled
  - startup preflight: `python scripts/parser_health.py --check`
  - immutable parser artifacts under `/app/parser-models`, selected through
    `DOCMIND_OCR__MODEL_CACHE_DIR` and kept outside the `/app/cache` volume
  - recurring liveness: `python scripts/container_health.py`, which checks only the local Streamlit TCP port
- Update `docker-compose.yml`:
  - use canonical `DOCMIND_*` env names
  - do not hardcode secrets
  - define healthcheck(s) (either container-level or compose-level)
  - run Ollama on CPU by default and set `DOCMIND_MODEL` to `qwen3:4b-instruct`
  - pin Ollama to `0.31.2` and Qdrant to `v1.18.2`
  - wait for Ollama and Qdrant readiness before starting DocMind
  - keep Ollama internal-only and connect through `DOCMIND_OLLAMA_BASE_URL=http://ollama:11434`
    - Note: `ollama` resolves to a private RFC1918 address on the compose network. When
      `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false`, DocMind rejects hostnames that
      resolve to private/link-local ranges (SSRF/DNS-rebinding hardening). For this
      deployment either set `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` or use a
      strict localhost architecture (shared network namespace) so DocMind connects via
      `http://localhost`.
    - if `DOCMIND_OLLAMA_ENABLE_WEB_SEARCH=true`, add `https://ollama.com` to
      `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` and enable remote endpoints explicitly
- Use uv's default `cpu` dependency group and explicit official PyTorch CPU
  index during `uv sync --frozen`. The app image therefore needs neither a
  custom wheel downloader nor CUDA-package exclusion lists.
- Add a hardened compose override (`docker-compose.prod.yml`) that sets
  `read_only: true` and `tmpfs` for `/tmp` while keeping `/app/cache` and
  `/app/logs` as volumes (operators can switch those to `tmpfs` if desired).
- Add `docker-compose.gpu.yml` with only Ollama's NVIDIA device reservation.
- Smoke-test Qdrant `v1.18.2` in hosted CI with the client version from
  `uv.lock`, requiring RRF and DBSF query responses.

## Security & Privacy

- No secrets should be baked into images.
- `.env` must be excluded by `.dockerignore` by default.
- Compose should allow passing `.env` at runtime only when desired.
- Remote endpoints remain blocked by default; containers should not change this posture.
- The bundled Ollama service should not be exposed publicly by default.
- The GPU override requires Docker Compose v2 and the NVIDIA Container Toolkit.

## Consequences

### Positive Outcomes

- Docker artifacts become runnable and aligned with repo constraints
- Reduced risk of secret leakage and root runtime
- Supported GPU path without coupling the app image to CUDA/toolkit churn

### Trade-offs

- Slightly more verbose Dockerfile (multi-stage)
- Only one language model backend is bundled. Power users can still use vLLM or LM Studio externally, but Compose does not ship those services.
- Production disables Streamlit source watching; local development retains
  automatic reloads.

## Changelog

- 1.6 (2026-07-11): Install OpenCV's required `libgl1` and `libglib2.0-0`
  runtimes and prove the baked RapidOCR artifacts through the startup
  preflight. Retain the 225 MB Mesa/X11 layer because the smaller uv exclusion
  leaves installed dependency metadata inconsistent.
- 1.5 (2026-07-11): Run CPU Ollama by default, move NVIDIA access to a minimal override, split parser startup readiness from TCP liveness, and pin a fusion-tested Qdrant server.
- 1.1 (2026-01-10): Add “single bundled GPU backend via profile” decision; keep app image CPU-only.
