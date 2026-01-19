# syntax=docker/dockerfile:1.7

FROM python:3.13.11-slim-bookworm AS builder

WORKDIR /app

ARG TORCH_VERSION=2.8.0
ARG TORCH_WHEEL_URL=""
ARG TORCH_WHEEL_SHA256=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        libmagic1 \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.9.24

ENV UV_HTTP_TIMEOUT=3600 \
    UV_HTTP_RETRIES=20 \
    UV_CONCURRENT_DOWNLOADS=1 \
    UV_CACHE_DIR=/root/.cache/uv-docker \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCH_WHEEL_URL=${TORCH_WHEEL_URL} \
    TORCH_WHEEL_SHA256=${TORCH_WHEEL_SHA256}

COPY pyproject.toml uv.lock README.md ./
COPY app.py ./app.py
COPY scripts/docker_fetch_torch_wheel.py /usr/local/bin/docker_fetch_torch_wheel.py

RUN --mount=type=cache,target=/root/.cache/torch \
    python3 /usr/local/bin/docker_fetch_torch_wheel.py

RUN --mount=type=cache,target=/root/.cache/uv-docker \
    --mount=type=cache,target=/root/.cache/torch \
    uv venv && \
    uv pip install --no-deps "$(cat /root/.cache/torch/torch-wheel.txt)" && \
    uv sync --frozen --no-dev --no-install-project \
      --no-install-package torch \
      --no-install-package nvidia-cublas-cu12 \
      --no-install-package nvidia-cuda-cupti-cu12 \
      --no-install-package nvidia-cuda-nvrtc-cu12 \
      --no-install-package nvidia-cuda-runtime-cu12 \
      --no-install-package nvidia-cudnn-cu12 \
      --no-install-package nvidia-cudnn-frontend \
      --no-install-package nvidia-cufft-cu12 \
      --no-install-package nvidia-cufile-cu12 \
      --no-install-package nvidia-curand-cu12 \
      --no-install-package nvidia-cusolver-cu12 \
      --no-install-package nvidia-cusparse-cu12 \
      --no-install-package nvidia-cusparselt-cu12 \
      --no-install-package nvidia-cutlass-dsl \
      --no-install-package nvidia-ml-py \
      --no-install-package nvidia-nccl-cu12 \
      --no-install-package nvidia-nvjitlink-cu12 \
      --no-install-package nvidia-nvtx-cu12

COPY src ./src
COPY templates ./templates


FROM python:3.13.11-slim-bookworm AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin docmind

COPY --from=builder --chown=docmind:docmind /app /app
RUN mkdir -p /app/data /app/cache /app/logs && \
    chown -R docmind:docmind /app/data /app/cache /app/logs

USER docmind

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import socket; s = socket.create_connection(('127.0.0.1', 8501), timeout=3); s.close()"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
