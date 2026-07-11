---
spec: SPEC-044
title: Runtime and toolchain baseline
version: 1.6.0
date: 2026-07-11
owners: ["ai-arch"]
status: Implemented
related_requirements:
  - NFR-MAINT-002: Ruff and Pyright pass.
  - NFR-MAINT-003: Package, lock, tests, and active docs agree.
  - NFR-PORT-003: Docker and wheel artifacts are reproducible.
related_adrs: ["ADR-065", "ADR-014", "ADR-024", "ADR-042"]
---

## Objective

This specification defines Python support, uv dependency resolution, CPU and GPU profiles, wheel validation, continuous integration (CI), and container baselines.

## Supported Python versions

- Primary development, CI, and container runtime: CPython 3.12.13
- Supported package range: Python `>=3.12,<3.14`
- Ruff target: `py312`
- Pyright target: Python 3.12
- Free-threaded CPython builds: unsupported

Verify the active interpreter:

```bash
uv sync --frozen
uv run python -c "import sys; print(sys.version)"
```

## Dependency authority

`pyproject.toml` declares direct dependencies, optional extras, dependency groups, and uv source rules. `uv.lock` records the resolved graph.

`tool.uv.required-version` pins uv `0.11.21` for local commands and
`astral-sh/setup-uv`; the Dockerfile uses the same versioned image. CI cache
suffixes separate the 3.12, 3.13, Qdrant, and documentation dependency payloads.
The isolated wheel backend is exact-pinned to Setuptools `80.9.0` and Wheel
`0.47.0`, so a manual rebuild of a release tag uses the same build toolchain.

The package requires:

- `llama-index-core>=0.14.21,<0.15.0`
- Selected LlamaIndex LLM, Hugging Face, Qdrant, and DuckDB adapters
- Direct FastEmbed, Sentence Transformers, Transformers, and Torch dependencies

The package does not include:

- The `llama-index` meta-package
- A `llama` optional extra
- LlamaIndex embedding adapters for OpenAI, FastEmbed, or CLIP
- Whisper

## CPU and GPU profiles

The default uv profile is CPU-first:

```bash
uv sync --frozen
uv run python -c \
  "import torch; assert torch.version.cuda is None; print(torch.__version__)"
```

The default `cpu` dependency group and uv source rules select official CPU-only PyTorch and Torchvision wheels.

The NVIDIA profile replaces the CPU group:

```bash
uv sync --frozen --no-group cpu --extra gpu
uv run python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The `cpu` group and `gpu` extra conflict by design. The GPU source rule selects
the locked CUDA 12.8 PyTorch wheel index without command-line index overrides.
It accelerates dense/image embeddings, reranking, and spaCy; sparse FastEmbed
remains on its mutually exclusive CPU distribution. Linux x86_64 is the
release-validated GPU host; WSL2 and macOS paths are best effort.

Other supported extras are:

- `multimodal`: ColPali reranking
- `observability`: LlamaIndex OpenTelemetry integration
- `eval`: BEIR evaluation
- `searchable-pdf`: POSIX-only OCRmyPDF integration (Linux, macOS, or WSL2;
  native Windows unsupported; OCRmyPDF and Tesseract executables required)

## Wheel contract

Build and inspect the wheel:

```bash
rm -rf build docmind_ai_llm.egg-info
uv build --wheel --clear
uv run python scripts/smoke_built_wheel.py
```

The smoke script verifies:

- Required parser, version, and prompt-template files
- Deleted files and any other paths absent from the source tree are absent
- Required `Requires-Dist` metadata
- Removed dependencies and the `graph` and `llama` extras are absent
- An isolated import and packaged prompt read after `pip install --no-deps`

This is a package content and metadata test. It does not validate a fresh pip dependency resolution.

uv is the supported local installer, and the repository image is the supported container path. The uv lock and source rules preserve the CPU or GPU profile. The wheel smoke test above validates package contents only; it does not define a supported pip-managed runtime.

## Continuous integration

The canonical CPython 3.12.13 lane:

- Set offline Hugging Face and Transformers flags
- Require real `llama_index.core`
- Assert `torch.version.cuda is None`
- Run Ruff, configured Pyright, wheel smoke, coverage, and schema validation

The lean CPython 3.13.12 lane installs the base profile, runs the full unit suite, and runs 21 focused GraphRAG tests. It asserts that `llama_index.core` is installed and Kuzu is absent. Independent jobs validate Docker and Compose policy, build the production image from its real context with reusable BuildKit layers in the GitHub Actions cache, require the canonical liveness command to succeed, and run RRF and DBSF queries against Qdrant `v1.18.2` with the qdrant-client version from `uv.lock`.

Workflows use the current supported Action majors: `actions/checkout@v7`, `actions/setup-python@v6`, `astral-sh/setup-uv@v8`, `actions/setup-node@v6`, `docker/setup-buildx-action@v4`, and `docker/build-push-action@v7`.

## Release workflow

The release workflow:

1. Creates the GitHub release through Release Please
2. Checks out the created release tag, clears ignored Setuptools staging, and
   builds its wheel with `uv build --wheel --clear`
3. Runs `scripts/smoke_built_wheel.py`
4. Uploads the smoke-tested `dist/*.whl` to the GitHub release

The workflow validates the release artifact’s contents and metadata. It does not publish to PyPI or claim dependency-install portability.

## vLLM policy

vLLM runs as an external OpenAI-compatible server. The application environment and container image do not install it.

Configure DocMind with:

- `DOCMIND_LLM_BACKEND=vllm`
- `DOCMIND_OPENAI__BASE_URL=http://localhost:8000/v1`
- `DOCMIND_OPENAI__API_KEY=local_api_key_not_used`

Server-side FlashInfer, FP8 key-value cache, model, and memory settings belong to the vLLM process.

## Container baseline

- Base image: `python:3.12.13-slim-bookworm`
- Installer: `uv sync --frozen`
- PyTorch profile: official CPU wheels
- Runtime user: non-root
- Command: exec-form `CMD`
- Framework telemetry: tracked Streamlit config and container command set
  `browser.gatherUsageStats=false`
- Startup preflight: `python scripts/parser_health.py --check`
- Immutable text model: pinned BGE-M3 PyTorch snapshot, offline-load checked at
  build time with a 1024-dimensional embedding assertion
- Recurring liveness: `python scripts/container_health.py`
- Compose base: CPU Ollama `0.31.2` and Qdrant `v1.18.2`
- Compose GPU mode: `docker-compose.gpu.yml` adds only NVIDIA device access

See SPEC-023 and ADR-042 for container hardening.

## Verification

Run the current local gates:

```bash
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run python scripts/run_tests.py --fast
rm -rf build docmind_ai_llm.egg-info
uv build --wheel --clear
uv run python scripts/smoke_built_wheel.py
```
