# Developer Setup for DocMind AI

## Prerequisites

- Python 3.9+.
- uv for package management.
- Git.
- Optional: Docker, NVIDIA CUDA for GPU testing.

## Installation

1. Clone:

   ```bash
   git clone https://github.com/BjornMelin/docmind-ai.git
   cd docmind-ai
   ```

2. Create venv and install:

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. For tests:

   ```bash
   uv sync --extra test
   ```

## Running Locally

```bash
streamlit run app.py
```

## Testing

```bash
pytest
```

## Linting/Formatting

```bash
ruff check .
ruff format .
```

## Environment Variables

See `.env.example` for configs like OLLAMA_BASE_URL.

For contributions, follow [CONTRIBUTING.md](../../CONTRIBUTING.md).
