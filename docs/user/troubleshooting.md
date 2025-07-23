# Troubleshooting DocMind AI

This guide helps resolve common issues when using DocMind AI.

## Common Issues and Solutions

### 1. Ollama Not Running

- **Symptoms**: "Connection refused" or "Cannot connect to Ollama" errors.
- **Solution**:
  1. Ensure Ollama is installed: `ollama --version`.
  2. Start Ollama: `ollama serve`.
  3. Verify URL: Default is `http://localhost:11434`. Update in sidebar if different.
  4. Check logs: `logs/app.log` for errors.

### 2. Dependency Installation Fails

- **Symptoms**: `uv sync` errors or missing packages.
- **Solution**:
  - Ensure Python 3.9+: `python --version`.
  - Update uv: `pip install -U uv`.
  - Retry: `uv sync` or `uv sync --extra gpu` for GPU support.
  - Check for conflicts: Use `uv pip list` to inspect installed versions.

### 3. GPU Not Detected

- **Symptoms**: GPU toggle ineffective; slow performance.
- **Solution**:
  - Verify NVIDIA drivers: `nvidia-smi`.
  - Install GPU dependencies: `uv sync --extra gpu`.
  - Ensure CUDA compatibility: Requires CUDA 12.x for FastEmbed.
  - Check VRAM: `utils.py:detect_hardware()` logs available VRAM.

### 4. Unsupported File Formats

- **Symptoms**: "Unsupported file type" error.
- **Solution**:
  - Supported formats: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB, code files.
  - Convert unsupported files (e.g., to PDF) before uploading.
  - Log issue on GitHub for new format support.

### 5. Analysis Errors

- **Symptoms**: "Error parsing output" or incomplete results.
- **Solution**:
  - Check document size: Enable chunking for large files.
  - Verify context size: Adjust in sidebar (e.g., 8192 for larger models).
  - Review raw output in `logs/app.log`.

### 6. Chat Interface Issues

- **Symptoms**: Irrelevant responses or retrieval failures.
- **Solution**:
  - Ensure vectorstore is created: Re-upload documents if needed.
  - Check hybrid search settings: Toggle multi-vector embeddings.
  - Review Qdrant logs: Ensure `QdrantClient` is running (`:memory:` mode).

## Getting Help

- Check `logs/app.log` for detailed errors.
- Search or open issues on [GitHub](https://github.com/BjornMelin/docmind-ai).
- Include: Steps to reproduce, logs, system details (OS, Python version, GPU).
