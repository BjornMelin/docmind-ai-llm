FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmupdf-dev=1.21.1+ds2-1+b4 \
    libmagic1=1:5.44-3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.14

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml uv.lock ./
ARG ENABLE_GPU=0
RUN uv venv "$VIRTUAL_ENV" && \
    if [ "$ENABLE_GPU" = "1" ]; then \
        uv sync --extra gpu --frozen; \
    else \
        uv sync --frozen; \
    fi

COPY . .

RUN python -m spacy download en_core_web_sm

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --silent --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
