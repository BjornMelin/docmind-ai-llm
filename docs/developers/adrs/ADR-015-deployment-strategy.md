---
ADR: 015
Title: Docker-First Local Deployment
Status: Accepted
Version: 5.1
Date: 2025-08-19
Supersedes:
Superseded-by:
Related: 004, 010, 011, 013, 024, 026
Tags: deployment, docker, local-first
References:
- [Docker Compose — Docs](https://docs.docker.com/compose/)
---

## Description

Provide a dead‑simple Docker deployment for the local Streamlit app. One image, one compose file, environment‑driven configuration, and optional GPU.

## Context

DocMind AI is a local app. Users should `docker compose up` and go. No multi‑stage builds or profiles by default; keep configuration in `.env`.

## Decision Drivers

- KISS and library‑first
- Works offline; minimal ops
- Clear, documented env vars

## Alternatives

- Multi‑profile, hardware‑auto builds — complex, brittle
- Bare‑metal scripts — inconsistent environments

### Decision Framework

| Option         | Simplicity (50%) | Reliability (30%) | Flex (20%) | Total | Decision    |
| -------------- | ---------------- | ----------------- | ---------- | ----- | ----------- |
| Single compose | 10               | 9                 | 7          | 9.2   | ✅ Selected |
| Multi‑profile  | 5                | 7                 | 9          | 6.3   | Rejected    |

## Decision

Publish a single Dockerfile and `docker-compose.yml`. Use env vars for model/provider flags, including FP8 + 128K (ADR‑004/010). GPU optional via NVIDIA toolkit.

## High-Level Architecture

Host `.env` → compose env → container app (`streamlit run app.py`)

## Related Requirements

### Functional Requirements

- FR‑1: Single command start
- FR‑2: Map `data/` and `models/` volumes

### Non-Functional Requirements

- NFR‑1: Image <2GB base; startup <45s (cached)

### Performance Requirements

- PR‑1: Cold start <60s; warm <45s on NVMe SSD
- PR‑2: Healthcheck returns 200 within 10s of Streamlit boot

### Integration Requirements

- IR‑1: `.env` drives runtime configuration
- IR‑2: Optional NVIDIA toolkit enables GPU

## Design

### Architecture Overview

- Single docker-compose with local services; environment-driven configuration

### Implementation Details

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY . .
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
```

```yaml
# docker-compose.yml
services:
  docmind:
    build: .
    ports: ["8501:8501"]
    volumes: ["./data:/app/data", "./models:/app/models"]
    environment:
      - VLLM_MAX_MODEL_LEN=131072
      - VLLM_KV_CACHE_DTYPE=fp8_e5m2
      - VLLM_ATTENTION_BACKEND=FLASHINFER
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Configuration

- `.env` holds all runtime flags (model name, provider, context window)

## Testing

- Smoke: container builds; health endpoint returns 200
- Launch: app reachable on `localhost:8501`

## Consequences

### Positive Outcomes

- One‑command start; easy to support
- Environment‑driven flexibility

### Negative Consequences / Trade-offs

- Less tailored than per‑hardware profiles

### Dependencies

- Docker, Docker Compose, (optional) NVIDIA Container Toolkit

### Ongoing Maintenance & Considerations

- Pin base image; track security updates and rebuild regularly
- Keep compose minimal and documented; avoid adding services by default

## Changelog

- 5.1 (2025‑09‑04): Standardized to template; added PR/IR
- 5.0 (2025‑08‑19): Accepted simplified Docker‑first approach
