---
ADR: 042
Title: Containerization Hardening (Dockerfile + Compose) with Optional GPU Backend (Ollama)
Status: Implemented
Version: 1.1
Date: 2026-01-10
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

Replace the current broken/inconsistent Docker artifacts with a **Python 3.11** `uv`-based, multi-stage Dockerfile and a compose configuration that:

- uses canonical `DOCMIND_*` env vars
- is secure-by-default (non-root, `.dockerignore`, no baked secrets)
- supports an **optional** GPU-capable local LLM backend via a **single** compose profile (`gpu`) using **Ollama** as the officially bundled GPU service (internal network; no host port publish by default)

## Context

The repository currently ships `Dockerfile` and `docker-compose.yml`, but they are not aligned with repo constraints:

- `Dockerfile` uses Python 3.12 (repo requires `<3.12`)
- Docker CMD is invalid (shell string inside JSON form)
- entrypoint path does not match Streamlit entry (`src/app.py`)
- `docker-compose.yml` uses non-canonical env variable names (not `DOCMIND_*`)
- `.dockerignore` is missing, increasing secret leak risk

Additionally, the “final release” posture requires a clear GPU story without forcing CUDA/PyTorch stacks into the app container. Bundling multiple GPU servers in-compose is high-maintenance; bundling none is a poor operator story.

## Decision Drivers

- Correctness: containers must run out-of-the-box
- Reproducibility: deterministic dependency install via `uv.lock`
- Security: non-root runtime, .dockerignore, avoid leaking `.env`
- Maintainability: keep Docker artifacts minimal and aligned with repo config discipline
- GPU capability: provide a supported local GPU path without coupling the app image to CUDA/toolkit churn

## Alternatives

- A: Minimal patch existing Dockerfile/compose (still leaves security/size debt; unclear GPU story)
- B: Single Dockerfile + compose with multiple GPU backends bundled (Ollama + vLLM) (more capability, but high support burden)
- C: GPU-enabled DocMind app image (CUDA/PyTorch inside) (brittle; huge; couples releases to GPU stack)
- D: Best-practice app container + **single bundled GPU backend (Ollama) via profile** (Selected)

### Decision Framework (≥9.0)

Weights: Complexity 40% · Perf/Size 30% · Alignment/Security 30% (10 = best)

| Option | Complexity (40%) | Perf/Size (30%) | Alignment/Security (30%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| **D: App container + single GPU backend profile (Ollama)** | 9.5 | 8.5 | 9.5 | **9.2** | ✅ Selected |
| B: App container + multiple GPU backends bundled | 6.5 | 7.5 | 7.5 | 7.1 | Rejected |
| C: GPU-enabled app image | 5.5 | 6.0 | 7.0 | 6.1 | Rejected |
| A: Minimal patch only | 8.0 | 5.0 | 6.0 | 6.7 | Rejected |

## Decision

Implement a ship-ready container baseline:

- Add `.dockerignore`
- Replace Dockerfile with:
  - Python 3.11 base
  - multi-stage build with `uv sync --frozen`
  - `UV_PYTHON_DOWNLOADS=never` to force uv to use container Python
  - non-root runtime user
  - correct Streamlit entrypoint: `streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501`
- Update `docker-compose.yml`:
  - use canonical `DOCMIND_*` env names
  - do not hardcode secrets
  - define healthcheck(s) (either container-level or compose-level)
  - include a `gpu` profile that starts **Ollama** with GPU access when available:
    - internal-only network (no `ports:` by default)
    - DocMind connects via `DOCMIND_OLLAMA_BASE_URL=http://ollama:11434`
    - extend `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` to include `http://ollama`
    - if Ollama Cloud web tools are enabled, allowlist `https://ollama.com` and
      enable remote endpoints explicitly
- Make `torch` installation reliable during container builds by prefetching the
  exact wheel with retry/resume and installing it before `uv sync --frozen`.
  Provide `TORCH_VERSION`, `TORCH_WHEEL_URL`, and optional `TORCH_WHEEL_SHA256`
  build args for overrides/verification. The fetch step defaults to the
  PyTorch CPU wheel index when available and falls back to PyPI.
- Keep the app image CPU-oriented by skipping CUDA-specific `nvidia-*` packages
  during `uv sync` (GPU inference remains external via the Ollama service).
- Add a hardened compose override (`docker-compose.prod.yml`) that sets
  `read_only: true` and `tmpfs` for `/tmp` while keeping `/app/cache` and
  `/app/logs` as volumes (operators can switch those to `tmpfs` if desired).

## Security & Privacy

- No secrets should be baked into images.
- `.env` must be excluded by `.dockerignore` by default.
- Compose should allow passing `.env` at runtime only when desired.
- Remote endpoints remain blocked by default; containers should not change this posture.
- The optional GPU backend service should not be exposed publicly by default (internal network only).
- GPU reservations require Docker Compose v2+; older compose variants may ignore
  `deploy.resources`. See SPEC-023 for a legacy `device_requests` snippet.

## Consequences

### Positive Outcomes

- Docker artifacts become runnable and aligned with repo constraints
- Reduced risk of secret leakage and root runtime
- Supported GPU path without coupling the app image to CUDA/toolkit churn

### Trade-offs

- Slightly more verbose Dockerfile (multi-stage)
- Only one GPU backend is bundled (Ollama). Power users can still use vLLM/LM Studio externally via configuration, but those services are not shipped as compose services.
- Runtime uses `libmupdf-dev` on Debian bookworm because a standalone runtime library package is not available; revisit if packaging changes.

## Changelog

- 1.1 (2026-01-10): Add “single bundled GPU backend via profile” decision; keep app image CPU-only.
