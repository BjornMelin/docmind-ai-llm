# Run DocMind locally

This guide installs the locked CPU profile, starts loopback-only services, prefetches parser models, configures a language model backend, and launches DocMind.

## Prerequisites

- Python 3.12 or 3.13
- uv
- Docker for the local Qdrant launcher
- Ollama or another OpenAI-compatible language model server

A GPU is optional. The default installation uses official CPU-only PyTorch
wheels. Linux x86_64 is the release-validated host platform; WSL2 and macOS are
best-effort paths until dedicated CI exists.

## Install the locked CPU profile

```bash
uv sync --frozen
uv run python -c \
  "import torch; assert torch.version.cuda is None; print(torch.__version__)"
```

## Install the optional NVIDIA profile

The GPU extra replaces the default CPU dependency group with locked CUDA 12.8
PyTorch wheels for dense/image embeddings, reranking, and spaCy. Sparse
FastEmbed remains CPU-based:

```bash
uv sync --frozen --no-group cpu --extra gpu
uv run python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

WSL2 is the best-effort Windows CUDA path; native Windows is not
release-validated. vLLM remains an external server and is not installed into
DocMind's environment.

## Choose a supported installation path

Use the locked uv environment for local development or the repository Docker
image for containers. DocMind does not publish a supported Python wheel or
library API.

## Start Qdrant

The launcher binds Qdrant REST and gRPC ports to loopback and refuses to reuse a container with missing or unsafe port bindings:

```bash
./scripts/start_qdrant_local.sh
curl --fail http://127.0.0.1:6333/readyz
uv run python scripts/qdrant_schema.py check
```

Docker Compose keeps Qdrant internal to the Compose network. Use the launcher when the host application or system tests need `127.0.0.1:6333`.

## Prefetch required local models

DocMind requires BGE-M3, BM42, the BGE reranker, and SigLIP for the default
retrieval pipeline. PDF parsing requires the verified Docling layout bundle.
RapidOCR models are packaged in its locked wheel:

```bash
uv run python tools/models/pull.py \
  --all \
  --cache_dir ./models_cache \
  --parser-defaults \
  --parser-cache-dir ./cache/models
uv run python scripts/parser_health.py --check
```

After dependencies and model bundles are present, the parser does not use a hosted fallback.

## Create local configuration

```bash
cp .env.example .env
```

Keep `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` for loopback-only backends.

### Configure Ollama

```bash
echo 'DOCMIND_LLM_BACKEND=ollama' >> .env
echo 'DOCMIND_OLLAMA_BASE_URL=http://localhost:11434' >> .env
echo 'DOCMIND_LLM_REQUEST__MODEL=qwen3:4b-instruct' >> .env
ollama pull qwen3:4b-instruct
```

### Configure an external vLLM server

```bash
echo 'DOCMIND_LLM_BACKEND=vllm' >> .env
echo 'DOCMIND_VLLM_BASE_URL=http://localhost:8000/v1' >> .env
echo 'DOCMIND_OPENAI__API_KEY=local_api_key_not_used' >> .env
```

vLLM runs outside DocMind. Configure its model, FlashInfer, FP8 cache, and memory limits on the server process.

### Configure an approved remote endpoint

Remote endpoints change the privacy boundary and require an explicit policy change:

```bash
echo 'DOCMIND_LLM_BACKEND=openai_compatible' >> .env
echo 'DOCMIND_OPENAI__BASE_URL=https://api.openai.com/v1' >> .env
echo 'DOCMIND_OPENAI__API_KEY=your_api_key_here' >> .env
echo 'DOCMIND_OPENAI__API_MODE=responses' >> .env
echo 'DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true' >> .env
```

## Launch DocMind

```bash
uv run streamlit run app.py
```

## Check the active runtime

```bash
uv run python -c \
  "from src.config import settings; print(settings.llm_backend)"
uv run python scripts/parser_health.py --check
uv run python scripts/qdrant_schema.py check
```

## Continue setup

- Configuration reference: `docs/user/configuration.md`
- Developer configuration: `docs/developers/configuration.md`
- Troubleshooting: `docs/user/troubleshooting-faq.md`
- Operations: `docs/developers/operations-guide.md`
