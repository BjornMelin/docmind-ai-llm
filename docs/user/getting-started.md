# Getting Started with DocMind AI

## Prerequisites

- Ollama installed (download from [ollama.com](https://ollama.com/)).

- Python 3.9+.

- Optional: Docker, NVIDIA GPU drivers.

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/BjornMelin/docmind-ai.git
   cd docmind-ai
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. For GPU:

   ```bash
   uv sync --extra gpu
   ```

## Running the App

- Locally:

  ```bash
  streamlit run src/app.py
  ```

- Docker:

  ```bash
  docker-compose up --build
  ```

Access at <http://localhost:8501>. Pull models via Ollama (e.g., `ollama pull qwen2:7b`).

## First Steps

1. Select backend/model in sidebar.
2. Upload documents.
3. Configure analysis options.
4. Click "Extract and Analyze".
5. Chat for follow-ups.

For detailed usage, see [usage-guide.md](usage-guide.md).
