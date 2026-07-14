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

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

ENV UV_HTTP_TIMEOUT=3600 \
    UV_HTTP_RETRIES=20 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_CACHE_DIR=/root/.cache/uv-docker \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_CACHE=/app/hf-models \
    DOCMIND_EMBEDDING__CACHE_FOLDER=/app/hf-models \
    LLAMA_INDEX_CACHE_DIR=/app/hf-models \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv-docker \
    uv sync --frozen --no-dev --no-install-project

COPY tools/models/pull.py ./tools/models/pull.py
COPY src/config/embedding_defaults.py ./src/config/embedding_defaults.py
COPY src/processing/parsing ./src/processing/parsing

# Persist partial Hub downloads across interrupted builds, then copy all
# default retrieval models into the immutable image. Keep this expensive layer
# separate from parser artifacts and unrelated application source.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python tools/models/pull.py \
      --all \
      --cache_dir /root/.cache/huggingface/hub \
    && mkdir -p /app/hf-models \
    && cp -a /root/.cache/huggingface/hub/. /app/hf-models/

# The production image advertises strict-offline PDF support, so the app-owned
# Docling layout bundle is fetched and verified during the build.
RUN python tools/models/pull.py \
      --parser-defaults \
      --parser-cache-dir /app/parser-models

# RapidOCR owns the default ONNX files shipped in its locked wheel. Prove the
# wheel is self-contained and the native defaults can perform real inference
# without a runtime download.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN --network=none python - <<'PY'
from pathlib import Path

import cv2
import numpy as np

from src.processing.parsing.backends.rapidocr_backend import run_rapidocr

path = Path("/tmp/rapidocr-offline.png")
image = np.full((220, 900, 3), 255, dtype=np.uint8)
cv2.putText(
    image,
    "DOCMIND OCR",
    (35, 145),
    cv2.FONT_HERSHEY_SIMPLEX,
    2.6,
    (0, 0, 0),
    6,
    cv2.LINE_AA,
)
if not cv2.imwrite(str(path), image):
    raise SystemExit("failed to write the RapidOCR smoke fixture")
if "DOCMIND OCR" not in run_rapidocr(path):
    raise SystemExit("RapidOCR packaged-model inference failed")
path.unlink()
PY

COPY README.md app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY src ./src
COPY scripts/container_entrypoint.sh scripts/container_health.py scripts/parser_health.py ./scripts/

RUN --network=none HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
from pathlib import Path

import torch
from fastembed import SparseTextEmbedding
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, TextNode

from src.config.embedding_defaults import DEFAULT_BM42_MODEL_ID
from src.config.integrations import setup_llamaindex
from src.retrieval.reranking import build_text_reranker
from src.utils.vision_siglip import load_siglip, siglip_features
from tools.models.pull import resolve_bm42_snapshot

setup_llamaindex(force_llm=True, force_embed=True)
embedding = getattr(Settings, "_embed_model", None)
if embedding is None or type(embedding).__name__ == "MockEmbedding":
    raise SystemExit("canonical BGE-M3 embedding did not load from the image")
if len(embedding.get_text_embedding("docmind readiness")) != 1024:
    raise SystemExit("canonical BGE-M3 embedding dimension is not 1024")

bm42_snapshot = resolve_bm42_snapshot(
    Path("/app/hf-models"),
    local_files_only=True,
)
bm42 = SparseTextEmbedding(
    model_name=DEFAULT_BM42_MODEL_ID,
    cache_dir="/app/hf-models",
    providers=["CPUExecutionProvider"],
    cuda=False,
    local_files_only=True,
    specific_model_path=bm42_snapshot,
)
document_sparse = next(bm42.embed("offline document readiness"))
query_sparse = next(bm42.query_embed("readiness"))
if not len(document_sparse.indices) or not len(query_sparse.indices):
    raise SystemExit("canonical BM42 document/query inference failed")

reranker = build_text_reranker(top_n=1)
if type(reranker).__name__ == "NoOpTextReranker":
    raise SystemExit("canonical BGE reranker did not load from the image")
reranked = reranker.postprocess_nodes(
    [NodeWithScore(node=TextNode(text="offline readiness"), score=1.0)],
    query_str="readiness",
)
if len(reranked) != 1:
    raise SystemExit("canonical BGE reranker inference failed")

siglip, processor, _device = load_siglip(device="cpu")
inputs = processor(
    text=["offline readiness"],
    padding="max_length",
    return_tensors="pt",
)
with torch.no_grad():
    features = siglip_features(siglip.get_text_features(**inputs))
if getattr(features, "shape", (0,))[0] != 1:
    raise SystemExit("canonical SigLIP inference failed")
PY


FROM python:3.12.13-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOCMIND_EMBEDDING__CACHE_FOLDER=/app/hf-models \
    DOCMIND_PARSING__MODEL_CACHE_DIR=/app/parser-models \
    HF_HUB_CACHE=/app/hf-models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
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
