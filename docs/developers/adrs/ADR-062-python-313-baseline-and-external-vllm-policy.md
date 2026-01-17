---
ADR: 062
Title: Python 3.13.11 Baseline (3.11–3.13 Supported) and External vLLM Policy
Status: Implemented
Version: 1.0
Date: 2026-01-16
Supersedes:
Superseded-by:
Related: 004, 024, 042
Tags: python, packaging, tooling, compatibility, llm
References:
  - https://www.python.org/downloads/release/python-31311/
  - https://docs.python.org/3/whatsnew/3.12.html
  - https://docs.python.org/3/whatsnew/3.13.html
  - https://devguide.python.org/versions/
  - https://pypi.org/project/vllm/0.10.1/
  - https://docs.vllm.ai/en/stable/contributing/
  - https://pypi.org/project/torch/2.7.1/
---

## Description

Adopt **CPython 3.13.11** as the primary runtime while keeping the application compatible with **Python 3.11–3.13** (`requires-python = ">=3.11,<3.14"`). Treat **vLLM as an external OpenAI-compatible server** (out-of-process) and do not require `vllm` as an in-repo dependency.

## Context

DocMind relies on a mix of pure-Python libraries and compiled dependencies (Torch, tokenizers, OCR/PDF tooling). The repository previously pinned Python 3.11, which blocked adoption of Python 3.13 features and security posture improvements.

During the Python 3.13 migration, a known blocker was encountered: the previously pinned `vllm` line was not installable on Python 3.13. Additionally, the multimodal lane (ColPali) constrained Torch versions, making an in-process vLLM path a high-risk coupling point for the lock.

The project is offline-first and local-first. vLLM and other inference servers already communicate via HTTP; keeping them out-of-process reduces resolver complexity, avoids native build churn, and preserves reproducibility across optional dependency lanes.

## Decision Drivers

- Reproducibility: `uv.lock` must resolve deterministically across supported runtimes.
- Operability: keep the default install CPU-first; GPU installs are explicit and documented.
- Maintainability: avoid a multi-stack matrix of mutually incompatible compiled deps inside a single environment.
- Performance: keep a fast path available (vLLM server profile) without binding the app to vLLM packaging constraints.
- Security posture: preserve offline-first defaults and avoid hidden runtime downloads.

## Alternatives

### Python Runtime Baseline

- A: Python 3.13 only (drop 3.11/3.12).
- B: Python 3.13.11 primary, support 3.11–3.13 via `>=3.11,<3.14`. (Selected)
- C: Keep Python 3.11 primary; treat 3.13 as optional.

### vLLM Integration Mode

- A: In-process vLLM (`pip install vllm` inside the repo env).
- B: External vLLM server only (OpenAI-compatible HTTP). (Selected)
- C: Remove vLLM support entirely and require Ollama only.

### Tooling Targeting (Ruff/Pyright)

- A: Target Python 3.13 in Ruff/Pyright (maximize modernization; risk accidental 3.11/3.12 breakage).
- B: Target Python 3.11 in Ruff/Pyright while running CI across 3.11–3.13. (Selected)
- C: Maintain multiple tooling configs per Python version (higher cognitive load; drift risk).

### Decision Framework (≥9.0)

Weights: Complexity 40% · Performance/Scale 30% · Ecosystem Alignment 30% (10 = best)

#### Python baseline

| Option | Complexity (40%) | Perf/Scale (30%) | Alignment (30%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| **B: 3.13.11 primary, support 3.11–3.13** | 9.0 | 9.0 | 9.5 | **9.2** | ✅ Selected |
| A: 3.13 only | 8.0 | 9.0 | 7.5 | 8.2 | Rejected |
| C: 3.11 primary | 8.5 | 7.0 | 7.0 | 7.7 | Rejected |

#### vLLM integration

| Option | Complexity (40%) | Perf/Scale (30%) | Alignment (30%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| **B: External vLLM server only** | 9.5 | 9.0 | 9.0 | **9.2** | ✅ Selected |
| A: In-process vLLM | 5.5 | 9.5 | 6.5 | 6.9 | Rejected |
| C: Remove vLLM support | 9.0 | 6.5 | 7.5 | 7.9 | Rejected |

#### Tooling targeting

| Option | Complexity (40%) | Perf/Scale (30%) | Alignment (30%) | Total | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| **B: Ruff/Pyright target 3.11; CI runs 3.11–3.13** | 9.5 | 8.0 | 9.5 | **9.0** | ✅ Selected |
| A: Target 3.13 only | 8.0 | 8.5 | 7.5 | 8.0 | Rejected |
| C: Multi-config tooling per version | 6.0 | 7.5 | 7.0 | 6.8 | Rejected |

## Decision

1) Set the repository’s primary runtime to **CPython 3.13.11** and keep compatibility with **3.11–3.13**.

2) Keep vLLM **out-of-process** and connect through OpenAI-compatible HTTP. The repo’s Python environment must not require `vllm` to install or run.

3) Configure Ruff/Pyright to target the **lowest supported runtime (3.11)** and enforce multi-version correctness via CI.

## Design

- `.python-version` pins `3.13.11` for deterministic local dev.
- `pyproject.toml` sets `requires-python = ">=3.11,<3.14"`.
- `.github/workflows/ci.yml` tests against 3.11, 3.12, and 3.13.11.
- `Dockerfile` pins `python:3.13.11-slim-bookworm`.
- GPU dependency lanes remain explicit (CUDA index specified on install) and are not globally applied.
- vLLM-specific knobs in `DOCMIND_VLLM__*` are treated as *server-launch helper inputs* (see `settings.get_vllm_env_vars()`); the app uses OpenAI-compatible HTTP at runtime.

## Consequences

### Positive outcomes

- Deterministic installs across multiple supported Python versions.
- Fewer lock failures and reduced native-build churn in the main app environment.
- vLLM performance path remains available without coupling the repo lock to vLLM packaging constraints.

### Trade-offs

- vLLM must be installed and operated separately (outside this repo env).
- Ruff/Pyright targeting 3.11 may avoid using some 3.13-only typing/stdlib features even when available.

## Changelog

- **1.0 (2026-01-16)**: Initial implemented version (Python 3.13.11 primary; external vLLM policy; multi-version CI).
