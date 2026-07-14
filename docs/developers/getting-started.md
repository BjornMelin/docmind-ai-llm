# Set up DocMind development

This guide creates the locked development environment, starts local dependencies, validates parser models, and runs the application and test gates.

## Install prerequisites

Install these tools before you clone the repository:

- Python 3.12 or 3.13
- uv
- Git
- Docker when you need local Qdrant or container validation

A graphics processing unit (GPU) is optional. The default dependency profile
uses official CPU-only PyTorch wheels. Linux x86_64 is release-validated; WSL2
and macOS paths are best effort until dedicated CI exists.

## Create the locked environment

Clone the repository and sync the lockfile:

```bash
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm
uv sync --frozen
```

Confirm that the default environment is CPU-only:

```bash
uv run python -c \
  "import torch; assert torch.version.cuda is None; print(torch.__version__)"
```

Use the locked NVIDIA profile only on a compatible Linux x86_64 host. It
accelerates PyTorch-based dense/image embeddings, reranking, and spaCy; sparse
FastEmbed remains CPU-based:

```bash
uv sync --frozen --no-group cpu --extra gpu
uv run python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The `gpu` extra and default `cpu` group conflict by design. uv selects the official CUDA 12.8 or CPU wheel index from `pyproject.toml`.

## Start loopback-only Qdrant

Use the repository launcher when the host application needs Qdrant on `127.0.0.1:6333`:

```bash
./scripts/start_qdrant_local.sh
uv run python scripts/qdrant_schema.py check
```

The launcher refuses to reuse an existing container unless its REST and gRPC ports match the expected loopback bindings. It preserves unsafe containers for operator inspection.

Docker Compose keeps Qdrant inside its network. Use Compose when you run the application as a container:

```bash
docker compose up -d qdrant
```

## Prefetch and verify required local models

The default retrieval pipeline requires BGE-M3, BM42, the BGE reranker, and
SigLIP. PDF parsing requires the pinned Docling layout bundle. RapidOCR uses the
models packaged in its locked wheel:

```bash
uv run python tools/models/pull.py \
  --all \
  --cache_dir ./models_cache \
  --parser-defaults \
  --parser-cache-dir ./cache/models
uv run python scripts/parser_health.py --check
```

The health command checks parser imports and hashes every file in the app-owned
Docling manifest. A missing, extra, linked, truncated, or modified Docling file
blocks dependency readiness. RapidOCR's locked wheel owns its model files and
checksums; the test and image-build gates run a real offline OCR fixture.

The parser framework, parser profile, and optical character recognition (OCR) engine are fixed. Configure parser resource limits and OCR behavior, but do not treat those fixed literals as backend selectors.

## Configure a language model backend

Copy the typed configuration template:

```bash
cp .env.example .env
```

Use a loopback Ollama service for the default local backend:

```env
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434
DOCMIND_LLM_REQUEST__MODEL=qwen3:4b-instruct
```

```bash
ollama pull qwen3:4b-instruct
```

Use an external vLLM service through its OpenAI-compatible endpoint:

```env
DOCMIND_LLM_BACKEND=vllm
DOCMIND_VLLM_BASE_URL=http://localhost:8000/v1
DOCMIND_OPENAI__API_KEY=local_api_key_not_used
```

Configure vLLM model, cache, and hardware options on the server process. DocMind does not install or start vLLM.

An approved remote endpoint changes the privacy boundary:

```env
DOCMIND_LLM_BACKEND=openai_compatible
DOCMIND_OPENAI__BASE_URL=https://api.openai.com/v1
DOCMIND_OPENAI__API_KEY=your_api_key_here
DOCMIND_OPENAI__API_MODE=responses
DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true
```

Never commit `.env` or credentials.

## Run the application

Start Streamlit from the locked environment:

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`, upload a supported document, then query its indexed content.

## Run focused development checks

Run checks that cover the files you changed:

```bash
uv run ruff format --check path/to/file.py
uv run ruff check path/to/file.py
uv run pyright --threads 4 path/to/file.py
uv run pytest tests/path/to/test_file.py -q
```

Use the repository runners for broader gates:

```bash
uv run pytest tests/unit tests/integration -q --no-cov
uv run pytest tests/unit tests/integration -q --cov=src --cov-branch --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json --cov-fail-under=80 --junitxml=junit.xml
```

Before shipping a branch, run the full static and test gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
```

## Validate system boundaries

Run the explicit Qdrant system test when ingestion, retrieval, persistence, or schema behavior changes:

```bash
DOCMIND_RUN_SYSTEM=1 \
  uv run pytest tests/system/test_e2e_offline.py -q
```

Run the canonical container health command inside a running application container:

```bash
python scripts/container_health.py
```

The container command is a recurring liveness probe that opens a TCP connection
to Streamlit without issuing an HTTP request. The entrypoint runs parser
dependency and Docling-model checks once before Streamlit starts.

Regenerate the parser benchmark only after the working tree is frozen:

```bash
uv run python scripts/benchmark_parsing.py \
  --generate-minimal-fixtures \
  --repeat 3 \
  --output docs/benchmarks/parser-runtime-validation.json
```

The checked-in artifact uses schema 3 and records its clean source commit,
runtime identity, fixture hashes, and repeat-three determinism evidence. Treat
its measurements as a workstation-specific regression baseline.

## Diagnose common setup failures

Use these checks before changing configuration:

```bash
uv run python -c "from src.config import settings; print(settings.llm_backend)"
uv run python scripts/parser_health.py --check
uv run python scripts/qdrant_schema.py check
```

If CUDA is unavailable, confirm the driver with `nvidia-smi`, then repeat the locked GPU sync. If Qdrant reports an incompatible schema, inspect it with `scripts/qdrant_schema.py check`. Never delete or rebuild a nonempty collection automatically.

## Continue development

Read the owning references before changing a subsystem:

- [Architecture overview](architecture-overview.md)
- [System architecture](system-architecture.md)
- [Configuration reference](configuration.md)
- [Operations guide](operations-guide.md)
- [Testing guide](../testing/testing-guide.md)
- [Architecture decisions](adrs/README.md)
