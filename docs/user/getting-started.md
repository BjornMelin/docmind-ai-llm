# Getting Started (User Guide)

DocMind AI is a local-first document analysis app. It runs offline after initial install and model downloads.

## Prerequisites

- **Python**: 3.13.11 recommended (supported: 3.11–3.13; see `pyproject.toml`)
- **uv**: dependency manager used by this repo
- **Docker** (optional but recommended): for running Qdrant locally
- **One local LLM backend**:
  - Ollama (default), or
  - an OpenAI-compatible server (vLLM / LM Studio / llama.cpp server)

## Step 1: Install dependencies

```bash
uv sync
```

### Optional: GPU extras (NVIDIA CUDA)

DocMind’s GPU extra accelerates embeddings/reranking and installs CUDA wheels for PyTorch via the PyTorch index.

```bash
uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Notes:

- On Windows, use WSL2 for CUDA installs.
- vLLM is not installed into this repo’s environment; see “vLLM (external server)” below.

## Step 2: Start local services (recommended)

Qdrant is the default vector database.

```bash
docker compose up -d qdrant
curl -f http://localhost:6333/health
```

## Step 3: Configure the app

Start from the canonical template and override only what you need:

```bash
cp .env.example .env
```

### Choose your LLM backend

DocMind uses a single `DOCMIND_*` configuration surface. Backends that expose an OpenAI-compatible API share the same client path.

- **Ollama (default)**:

  ```bash
  echo 'DOCMIND_LLM_BACKEND=ollama' >> .env
  echo 'DOCMIND_OLLAMA_BASE_URL=http://localhost:11434' >> .env
  ```

- **vLLM (external OpenAI-compatible server)**:

  ```bash
  echo 'DOCMIND_LLM_BACKEND=vllm' >> .env
  echo 'DOCMIND_OPENAI__BASE_URL=http://localhost:8000/v1' >> .env
  echo 'DOCMIND_OPENAI__API_KEY=not-needed' >> .env
  ```

For a full list of configuration knobs, see:

- `docs/user/configuration.md`
- `docs/developers/configuration.md`

## Step 4: Run DocMind

```bash
streamlit run src/app.py
```

## Quick diagnostics

```bash
uv run python -c "from src.config import settings; print(settings.llm_backend, settings.backend_base_url_normalized)"
uv run python -c "import torch; print('cuda', torch.cuda.is_available())"
```

## vLLM (external server)

DocMind connects to vLLM via OpenAI-compatible HTTP (`/v1`). Run vLLM separately (Linux + NVIDIA GPU recommended).

If you need a server-side FlashInfer/FP8 profile, see:

- `README.md` (vLLM server connectivity + tuning)
- `docs/developers/operations-guide.md`

## Next steps

- Configuration reference: `docs/user/configuration.md`
- Troubleshooting: `docs/user/troubleshooting-faq.md`
