# Troubleshooting & FAQ

This page combines quick fixes for common issues and answers to frequently asked questions.

## Quick Diagnostics

```bash
# System checks
nvidia-smi                          # GPU status
uv run python -c "import sys; print(sys.version)"  # Python 3.13.11
curl http://localhost:11434/api/version  # Ollama service

# App checks
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
uv run python -c "from src.config import settings; print('Config loaded OK')"
```

## Common Issues

### App won’t start

```bash
uv sync                 # Reinstall deps
uv run python -c "import sys; print(sys.version)"  # Should be 3.13.11
find . -name "*.pyc" -delete; find . -name "__pycache__" -type d -exec rm -rf {} +
lsof -i :8501          # Free port 8501 if in use
```

### GPU not detected

```bash
nvidia-smi
uv run python -c "import torch; print(torch.cuda.is_available())"
# Prefer explicit embed-device selection for embeddings + enable GPU acceleration
printf "\nDOCMIND_EMBEDDING__EMBED_DEVICE=cuda\nDOCMIND_ENABLE_GPU_ACCELERATION=true\n" >> .env
```

### Out of memory (CUDA)

```bash
printf "\nDOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.75\nDOCMIND_CONTEXT_WINDOW_SIZE=32768\n" >> .env
uv run python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

### Ollama connection failed

```bash
curl http://localhost:11434/api/version || ollama serve
ollama list || ollama pull qwen3-4b-instruct-2507
```

### Cache issues (document-processing cache)

- Location: `settings.cache_dir/docmind.duckdb`
- Clear cache: delete `docmind.duckdb` (recreated on next run)
- Move cache: set `DOCMIND_CACHE_DIR` and restart
- Permission error: choose a writable directory
- DB busy/lock: stop concurrent writers (single-writer DB)
- Corruption: stop app, delete file, restart

See also: ../developers/cache.md (Implementation Guide).

### Staleness badge shown in Chat

- The Chat page compares the current corpus/config hashes to the latest snapshot manifest. If they differ, a staleness badge appears.
- Fix: Open the Documents page and rebuild snapshots. After the rebuild, Chat will auto‑load the latest non‑stale snapshot.

### Hybrid retrieval results look odd

- Ensure Qdrant is reachable and collections have named vectors `text-dense` and `text-sparse` (sparse index with IDF modifier).
- Fusion is performed server‑side via the Qdrant Query API (RRF default; DBSF optional). There are no client‑side fusion knobs.

## FAQs

### Do I need internet?

Only for initial install and model downloads. The app runs offline after setup.

### What hardware is recommended?

- Minimum: 8GB RAM
- Recommended: 16GB RAM, RTX 4060 (12GB VRAM)
- Optimal: 32GB RAM, RTX 4090

### Can I run without a GPU?

Yes. It works on CPU, but responses are slower.

### Where are my documents stored?

Documents are processed locally. Results may be cached locally; nothing leaves your machine.

### What file types are supported?

PDF, DOCX, TXT, RTF, MD, XLSX, CSV, PPTX, JSON, XML, MSG, ODT, EPUB, HTML.

### How do I optimize performance?

If you run vLLM, configure performance options (FlashInfer, FP8 KV cache, chunked prefill) on the vLLM server process. See docs/user/configuration.md.

### How do I get help?

- docs/user/getting-started.md
- docs/user/configuration.md
- GitHub Issues: <https://github.com/BjornMelin/docmind-ai-llm/issues>
