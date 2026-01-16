---
title: GPU Setup Guide
version: 1.0.0
date: 2026-01-16
---

## GPU Setup Guide

This guide covers optional GPU acceleration for:

- **LLM serving** (vLLM)
- **Embeddings** (fastembed-gpu)
- **NLP enrichment** (spaCy via Thinc)

For baseline CPU install, see `docs/user/getting-started.md`.

## NVIDIA CUDA (Linux / Windows / WSL2)

### Prerequisites

- NVIDIA driver installed and working (`nvidia-smi`)
- CUDA 12.x toolkit available (`nvcc --version`) if required by your environment

### Install (GPU Extras)

Use the project’s GPU extras. This installs:

- `vllm` + `flashinfer-python` (LLM serving acceleration)
- `fastembed-gpu` (embedding acceleration)
- `cupy-cuda12x>=13` (spaCy/Thinc GPU acceleration for CUDA 12.x)

```bash
uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match
uv run python -m spacy download en_core_web_sm
```

### Runtime selection (spaCy / CUDA)

Canonical variables (recommended):

```bash
export DOCMIND_SPACY__DEVICE=auto   # cpu|cuda|apple|auto
export DOCMIND_SPACY__GPU_ID=0
```

Shorthand aliases (supported for backwards-compatibility via `_SPACY_ENV_BRIDGE`):

```bash
export SPACY_DEVICE=auto   # or cuda
export SPACY_GPU_ID=0
```

### Verify (spaCy / CUDA)

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "import spacy; print(spacy.__version__); print(spacy.prefer_gpu(0))"
```

## Apple Silicon (macOS arm64)

### Install (Apple Extras)

```bash
uv sync --extra apple
uv run python -m spacy download en_core_web_sm
```

### Runtime selection (spaCy / Apple)

```bash
export SPACY_DEVICE=auto   # or apple
```

### Verify (spaCy / Apple)

```bash
uv run python -c "import spacy; print(spacy.__version__); print(spacy.prefer_gpu(0))"
```

## Notes on spaCy CUDA extras

spaCy publishes optional CUDA extras like `spacy[cuda12x]`, but the project currently uses
`numpy==2.x` across platforms. spaCy’s CUDA 12.x extra pins `cupy-cuda12x<13`, which is
constrained to `numpy<1.29` and cannot be represented alongside the Apple acceleration stack
in a single `uv.lock` (one version per package).

DocMind therefore installs `cupy-cuda12x>=13` in the GPU extra to keep the lock solvable while
still enabling spaCy/Thinc GPU execution.
