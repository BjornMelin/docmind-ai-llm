# Developer Setup for DocMind AI

## Prerequisites

- Python 3.11+ (tested with 3.11, 3.12).

- uv for package management.

- Git.

- Optional: Docker, NVIDIA CUDA 12.8+ for GPU testing with RTX 4090.

## Installation

1. Clone:

   ```bash
   git clone https://github.com/BjornMelin/docmind-ai-llm.git
   cd docmind-ai-llm
   ```

2. Create venv and install:

   ```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```

3. For GPU development (RTX 4090 with vLLM FlashInfer):

   ```bash
   # Install PyTorch 2.7.1 with CUDA 12.8
   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
       --extra-index-url https://download.pytorch.org/whl/cu128
   
   # Install vLLM with FlashInfer
   uv pip install "vllm[flashinfer]>=0.10.1" \
       --extra-index-url https://download.pytorch.org/whl/cu128
   
   # Install GPU extras
   uv sync --extra gpu
   ```

4. For tests:

   ```bash
   uv sync --extra test
   ```

5. Install spaCy model:

   ```bash
   uv run python -m spacy download en_core_web_sm
   ```

## Running Locally

```bash
streamlit run src/app.py
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
