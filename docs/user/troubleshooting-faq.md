# Troubleshoot DocMind

Use these checks to isolate setup, parser, storage, model, and performance problems before changing application data.

## Run the first diagnostics

Check the locked environment and local services:

```bash
uv run python -c "import sys; print(sys.version)"
uv run python -c \
  "from src.config import settings; print(settings.llm_backend)"
uv run python scripts/parser_health.py --check
uv run python scripts/qdrant_schema.py check
```

## Fix application startup

Resync the locked environment and start Streamlit:

```bash
uv sync --frozen
uv run streamlit run app.py
```

If port 8501 is busy, inspect the owning process with `lsof -i :8501`. Stop that process or configure another Streamlit port.

## Fix missing parser models

Prefetch the pinned Docling and RapidOCR bundles, then validate every manifest entry:

```bash
uv run python tools/models/pull.py \
  --parser-defaults \
  --rapidocr-cache-dir cache/models
uv run python scripts/parser_health.py --check
```

A missing, extra, linked, truncated, or modified model file blocks PDF readiness. The health output reports affected paths relative to each model root.

## Fix Qdrant connection or schema errors

Start loopback-only Qdrant and inspect the collection:

```bash
./scripts/start_qdrant_local.sh
uv run python scripts/qdrant_schema.py check
```

Runtime startup creates a missing collection and refuses an incompatible existing one. Use `scripts/qdrant_schema.py rebuild-empty` only after stopping every writer and confirming that the collection contains zero points.

## Fix an Ollama connection

Confirm that Ollama is running and that the configured model exists:

```bash
curl --fail http://localhost:11434/api/version
ollama list
```

Start `ollama serve` or pull the configured model when either check fails.

## Fix an external vLLM connection

Confirm the application settings and query the server’s OpenAI-compatible model endpoint:

```bash
echo "$DOCMIND_LLM_BACKEND"
echo "$DOCMIND_OPENAI__BASE_URL"
curl --fail --silent "$DOCMIND_OPENAI__BASE_URL/models"
```

Configure FlashInfer, quantization, context, and memory limits on the vLLM server process. DocMind does not own those server settings.

## Fix missing GPU acceleration

The default environment is CPU-only. Replace its CPU group with the locked NVIDIA extra:

```bash
nvidia-smi
uv sync --frozen --no-group cpu --extra gpu
uv run python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use Windows Subsystem for Linux 2 (WSL2) for the best-effort Windows CUDA path;
native Windows is not release-validated. If CUDA remains unavailable, verify
that the installed driver supports the locked CUDA 12.8 wheels.

## Reduce memory pressure

Choose smaller models or context limits in the language model server first. Close unrelated GPU workloads and inspect current device use:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Document parsing works on CPU. GPU capacity depends on the selected embedding, reranking, and external language model workloads, so the project does not publish one hardware minimum.

## Repair the ingestion cache

Stop every application process before changing cache files. The default DuckDB ingestion cache lives under the configured `DOCMIND_CACHE__DIR` and `DOCMIND_CACHE__FILENAME`.

Back up the cache when its contents matter. Remove it only when you intend to discard cached ingestion results; DocMind recreates it on the next run.

## Rebuild stale snapshots

The Chat page compares current corpus and configuration hashes with the latest snapshot manifest. Rebuild snapshots from the Documents page when the staleness indicator appears.

## Interpret local-first privacy

Parsing, embedding, reranking, Qdrant, and persistence run locally by default. Initial dependency and model downloads access their configured upstream sources.

An explicitly configured remote language model endpoint or Ollama web tool sends requests outside the local machine. Keep remote endpoints disabled unless that trust-boundary change is approved.

## Run DocMind without a GPU

Use the default locked profile:

```bash
uv sync --frozen
```

CPU execution is supported. Measure response time on the target machine because model, context, document, and corpus sizes change runtime behavior.

## Measure performance

Run the repository monitor instead of relying on hardware-wide estimates:

```bash
uv run python scripts/performance_monitor.py \
  --run-tests \
  --check-regressions \
  --report
```

Record the model, context, hardware, corpus, command, and generated artifact with any result.

## Find more help

Use these references for the next diagnostic step:

- [Run DocMind locally](getting-started.md)
- [Configure DocMind](configuration.md)
- [Operate DocMind](../developers/operations-guide.md)
- [GitHub issues](https://github.com/BjornMelin/docmind-ai-llm/issues)
