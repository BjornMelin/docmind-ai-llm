# syntax=docker/dockerfile:1.7

FROM python:3.12.13-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libgomp1 \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

ENV UV_HTTP_TIMEOUT=3600 \
    UV_HTTP_RETRIES=20 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_CACHE_DIR=/root/.cache/uv-docker \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    DOCMIND_EMBEDDING__CACHE_FOLDER=/app/hf-models \
    LLAMA_INDEX_CACHE_DIR=/app/hf-models \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv-docker \
    uv sync --frozen --no-dev --no-install-project

COPY README.md app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY src ./src
COPY scripts/container_entrypoint.sh scripts/container_health.py scripts/parser_health.py ./scripts/
COPY tools/models/pull.py ./tools/models/pull.py

# The production image advertises strict-offline PDF support, so its required
# public parser artifacts are part of the immutable image rather than fetched
# on first use.
RUN python tools/models/pull.py \
      --bge-m3 \
      --cache_dir /app/hf-models \
      --parser-defaults \
      --rapidocr-cache-dir /app/parser-models

RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from llama_index.core import Settings

from src.config.integrations import setup_llamaindex

setup_llamaindex(force_llm=True, force_embed=True)
embedding = getattr(Settings, "_embed_model", None)
if embedding is None or type(embedding).__name__ == "MockEmbedding":
    raise SystemExit("canonical BGE-M3 embedding did not load from the image")
if len(embedding.get_text_embedding("docmind readiness")) != 1024:
    raise SystemExit("canonical BGE-M3 embedding dimension is not 1024")
PY


FROM python:3.12.13-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOCMIND_EMBEDDING__CACHE_FOLDER=/app/hf-models \
    DOCMIND_OCR__MODEL_CACHE_DIR=/app/parser-models \
    LLAMA_INDEX_CACHE_DIR=/app/hf-models \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin docmind

COPY --from=builder --chown=docmind:docmind /app /app
RUN mkdir -p /app/data /app/cache /app/logs && \
    chown -R docmind:docmind /app/data /app/cache /app/logs

USER docmind

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["python", "scripts/container_health.py"]

ENTRYPOINT ["./scripts/container_entrypoint.sh"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
