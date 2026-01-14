# Operations Guide

**Version**: 2.0 (Rock Solid)
**Scope**: Production, Offline, and Hardened Environments

## 1. Quick Start (Installation)

DocMind AI is a pure Python project managed by `uv`.

### Prerequisites

* **Python**: 3.11 (Managed by `uv`)
* **GPU**: NVIDIA RTX 4090 (16GB VRAM) recommended for vLLM/ColPali.
* **System**: Linux / MacOS / WSL2.

### Standard Install

```bash
# 1. Install dependencies & tools
uv sync --frozen

# 2. Run the application
uv run src/app.py
```

---

## 2. Infrastructure & Hardening

### Offline-First Setup

To operate in air-gapped or restricted environments, you must pre-download artifacts.

1. **Download Models**: Use the CLI to fetch LLM and embedding weights.

    ```bash
    uv run tools/models/pull.py --all
    ```

2. **Runtime Flags**: Set these environment variables to strictly prevent network egress.

    ```env
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    ```

3. **spaCy**: Ensure the `en_core_web_sm` model is installed locally.

    ```bash
    uv run python -m spacy download en_core_web_sm
    ```

### Container Hardening (ADR-042)

The provided `Dockerfile` supports hardened production security.

* **Non-Root User**: Application runs as a standard user (`uid=1000`).
* **Read-Only Filesystem**:
  * Set `read_only: true` in `docker-compose.prod.yml`.
  * Mount `tmpfs` at `/tmp` and `/run`.
  * Mount persistent volumes for `/app/data` and `/app/logs`.
* **Health Check**:
  * Endpoint: `/_stcore/health`
  * Port: `8501`

---

## 3. Backend Selection & Performance

Choose the backend that fits your hardware and throughput needs.

| Backend | Hardware | Use Case | Key Strength |
| :--- | :--- | :--- | :--- |
| **vLLM** | NVIDIA GPU (16GB+) | Production / High Load | Highest throughput; FlashInfer optimized. |
| **Ollama** | Mac / Linux / Win | General / Easy Setup | Zero-config; manages model lifecycle. |
| **LlamaCPP** | Local Disk / CPU | Low Resource / Edge | Runs `.gguf` directly; low overhead. |
| **LM Studio** | Desktop UI | Research / Dev | Visual parameter tuning. |

### Performance Tuning (vLLM)

To achieve the **128K context window** on 16GB VRAM, exact arguments are required:

* **Max Model Length**: `--max-model-len 131072`
* **KV Cache**: `--kv-cache-dtype fp8_e5m2` (Critical reduction in memory usage).
* **Memory Utilization**: `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.90`

```bash
# Production Launch Implementation
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill
```

---

## 4. Advanced Operations

### Multimodal Reranking

DocMind supports dual-path visual reranking for complex PDF/Image retrieval.

* **Default (SigLIP)**: Fast, zero-service cosine rescale. No extra VRAM cost (uses embedding model).
* **Pro (ColPali)**: Full visual-semantic reranking. Requires ~8-12GB additional VRAM.
  * **Enable**: `DOCMIND_RETRIEVAL__ENABLE_COLPALI=true`
  * **Warning**: Enabling this on 16GB cards alongside vLLM may cause OOM. Monitor VRAM.

### Agent Operations

* **Decision Timeout**: `DOCMIND_AGENTS__DECISION_TIMEOUT=200` (ms). Increase this if the router aggressively fallbacks during heavy load (e.g., to 400ms).

---

## 5. Maintenance & Data Protection

### Data Persistence

State is stored in three local locations (under `./data/`):

1. `docmind.db` (SQLite WAL): **Operational Metadata** (Jobs, Snapshots, UI State). ACID compliant.
2. `docmind.duckdb` (DuckDB): **Ingestion Cache**. Loss of this file slows down re-ingestion but loses no data.
3. `qdrant/` (Directory): **Vector Storage**. Contains the encoded index.

### Manual Backup Strategy

*Automation script is planned (ADR-033/051).*

**To Backup**:

1. **Stop the Application**.
2. **Copy Artifacts**:

    ```bash
    cp -r data/docmind.db backups/
    cp -r data/qdrant/ backups/
    # Optional: Cache can be rebuilt
    cp data/docmind.duckdb backups/
    ```

3. **Restart Application**.

### Log Safety & Observability

* **PII Policy**: Logs are automatically redacted (ADR-047). Sensitive fields are replaced with robust HMAC fingerprints using `DOCMIND_HASHING__HMAC_SECRET`.
* **Encrypted Artifacts**: If `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=true`, page images on disk are AES-256 encrypted using `DOCMIND_IMG_AES_KEY_BASE64`.

---

## 6. Configuration Guide

Variables are prefixed with `DOCMIND_`. Nested keys use `__`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DOCMIND_LLM_BACKEND` | `ollama` | `vllm`, `ollama`, or `llamacpp`. |
| `DOCMIND_VLLM_BASE_URL` | `http://localhost:8000/v1` | Target for vLLM backend. |
| `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION` | `0.85` | VRAM budget (0.0 to 1.0). |
| `DOCMIND_PROCESSING__CHUNK_SIZE` | `1024` | Tokens per document node. |
| `DOCMIND_PROCESSING__CHUNK_OVERLAP` | `100` | Overlap tokens. |
| `DOCMIND_RETRIEVAL__ENABLE_COLPALI` | `false` | Enable heavy visual reranker. |
| `DOCMIND_CONTEXT_WINDOW` | `131072` | Global context cap. |
| `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES` | `false` | Enable AES encryption. |
| `DOCMIND_IMG_AES_KEY_BASE64` | `None` | 32-byte key (Required if encrypt enabled). |

---

## 7. Developer Cheatsheet

Standard commands for repo maintenance.

| Task | Command |
| :--- | :--- |
| **Sync Deps** | `uv sync --frozen` |
| **Update Lock** | `uv lock` |
| **Lint/Format** | `uv run ruff check .` / `uv run ruff format .` |
| **Type Check** | `uv run pyright` |
| **Run Tests** | `uv run python scripts/run_tests.py` |
| **Quality Gate** | `uv run scripts/run_quality_gates.py` |
