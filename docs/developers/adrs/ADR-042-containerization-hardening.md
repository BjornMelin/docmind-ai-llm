---
ADR: 042
Title: Containerization Hardening (Dockerfile + Compose) for Ship-Ready Deployment
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 015, 024
Tags: docker, compose, packaging, security, deployment
References:
  - https://docs.docker.com/build/building/multi-stage/
  - https://docs.docker.com/compose/
---

## Description

Replace the current broken/inconsistent Docker artifacts with a **Python 3.11** `uv`-based, multi-stage Dockerfile and a compose configuration that uses canonical `DOCMIND_*` env vars.

## Context

The repository currently ships `Dockerfile` and `docker-compose.yml`, but they are not aligned with repo constraints:

- `Dockerfile` uses Python 3.12 (repo requires `<3.12`)
- Docker CMD is invalid (shell string inside JSON form)
- entrypoint path does not match Streamlit entry (`src/app.py`)
- `docker-compose.yml` uses non-canonical env variable names (not `DOCMIND_*`)
- `.dockerignore` is missing, increasing secret leak risk

This is a v1 ship blocker.

## Decision Drivers

- Correctness: containers must run out-of-the-box
- Reproducibility: deterministic dependency install via `uv.lock`
- Security: non-root runtime, .dockerignore, avoid leaking `.env`
- Maintainability: keep Docker artifacts minimal and aligned with repo config discipline

## Alternatives

- A: Minimal patch existing Dockerfile/compose (still leaves security/size debt)
- B: Adopt docker-architect best-practice templates (Selected)
- C: Remove Docker support for v1 (not acceptable since artifacts already exist)

### Decision Framework (≥9.0)

| Option                                                            | Complexity (40%) | Perf/Size (30%) | Alignment/Security (30%) |   Total |
| ----------------------------------------------------------------- | ---------------: | --------------: | -----------------------: | ------: |
| **B: Best-practice multi-stage + .dockerignore + canonical envs** |                9 |               9 |                       10 | **9.3** |
| A: Minimal patch                                                  |                8 |               5 |                        6 |     6.7 |
| C: Remove                                                         |               10 |               0 |                        2 |     5.2 |

## Decision

Implement a ship-ready container baseline:

- Add `.dockerignore`
- Replace Dockerfile with:
  - Python 3.11 base
  - multi-stage build with `uv sync --frozen`
  - non-root runtime user
  - correct Streamlit entrypoint: `streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501`
- Update `docker-compose.yml`:
  - use canonical `DOCMIND_*` env names
  - do not hardcode secrets
  - define healthcheck(s) (either container-level or compose-level)

## Security & Privacy

- No secrets should be baked into images.
- `.env` must be excluded by `.dockerignore` by default.
- Compose should allow passing `.env` at runtime only when desired.
- Remote endpoints remain blocked by default; containers should not change this posture.

## Consequences

### Positive Outcomes

- Docker artifacts become runnable and aligned with repo constraints
- Reduced risk of secret leakage and root runtime

### Trade-offs

- Slightly more verbose Dockerfile (multi-stage)

## Changelog

- 1.0 (2026-01-09): Proposed for v1 ship readiness.
