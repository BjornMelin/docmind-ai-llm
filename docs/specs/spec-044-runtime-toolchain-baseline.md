---
spec: SPEC-044
title: Runtime & Toolchain Baseline — Python 3.13.11 Only, uv-first
version: 1.1.0
date: 2026-01-17
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - NFR-MAINT-002: Ruff/pyright pass (ruff enforces pylint-equivalent rules).
  - NFR-MAINT-003: No placeholders; docs/specs/RTM must match code.
  - NFR-PORT-003: Docker/compose runnable + reproducible.
related_adrs: ["ADR-064", "ADR-014", "ADR-024", "ADR-042"]
---

## Objective

Define a single, enforceable baseline for:

- supported Python versions (primary vs. supported range)
- toolchain targeting (Ruff/Pyright)
- deterministic dependency resolution across extras/groups
- container and CI Python baselines
- uv-first developer workflows

## Non-goals

- Supporting Python <3.11.
- Adopting the free-threaded CPython build (GIL-disabled). The project uses the default CPython build.
- Installing or running GPU inference servers (vLLM/LM Studio) inside the app environment or container image.

## Runtime policy

### Supported versions

- Primary development/runtime version: **CPython 3.13.11** (see `.python-version`).
- Supported range for the application: **Python 3.13** (`requires-python = ">=3.13,<3.14"`).

```bash
uv sync --frozen
uv run python -c "import sys; print(sys.version)"
uv run python scripts/run_tests.py
```

## Tooling policy (lint + type)

DocMind targets Python 3.13 only:

- Ruff targets Python 3.13 (`target-version = "py313"`).
- Pyright checks against Python 3.13 (`pythonVersion = "3.13"`).
- CI runs the full test suite against Python 3.13.11.

## Dependency resolution policy (uv)

- Source of truth: `pyproject.toml` + `uv.lock`.
- Base install must remain CPU-first and resolve without CUDA-specific indices.
- Optional extras must be deterministic and documented:
  - `uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match`
  - `uv sync --extra graph`
  - `uv sync --extra multimodal`
  - `uv sync --extra observability`
  - `uv sync --extra eval`

## vLLM policy

vLLM is supported as an **external OpenAI-compatible server** (out-of-process). The repo environment does not install `vllm`.

DocMind connects via:

- `DOCMIND_LLM_BACKEND=vllm`
- `DOCMIND_OPENAI__BASE_URL=http://localhost:8000/v1` (or `DOCMIND_VLLM__VLLM_BASE_URL`)

Server tuning (FlashInfer, FP8 KV cache, chunked prefill) is configured on the vLLM server process. DocMind may surface helper values via `settings.get_vllm_env_vars()` for convenience.

## Container baseline

- Containers pin the primary runtime: `python:3.13.11-slim-bookworm`.
- Dependency install in containers uses `uv sync --frozen` against `uv.lock`.
- Runtime container runs as a non-root user and uses exec-form `CMD`.

See `SPEC-023` and `ADR-042`.

## Verification

Required checks for final validation:

```bash
uv run ruff format . && uv run ruff check . --fix
uv run pyright --threads 4
uv run python scripts/run_tests.py
uv run python scripts/run_quality_gates.py --ci --report
```
