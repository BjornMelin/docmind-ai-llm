# Deploying DocMind AI

This guide covers deployment options for DocMind AI, including local, Docker, and production setups.

## Local Deployment

1. Clone and install:

   ```bash
   git clone https://github.com/BjornMelin/docmind-ai.git
   cd docmind-ai
   uv sync
   ```

2. Run:

   ```bash
   streamlit run src/app.py
   ```

3. Access: <http://localhost:8501>.

## Docker Deployment

1. Ensure Docker and Docker Compose are installed.
2. Build and run:

   ```bash
   docker-compose up --build
   ```

3. Access: <http://localhost:8501>.
4. Notes:
   - Dockerfile installs dependencies via uv.
   - docker-compose.yml configures services (Streamlit, Qdrant, Ollama).
   - GPU support requires NVIDIA Container Toolkit.

## Production Deployment

- **Requirements**:
  - Server with Python 3.9+, optional NVIDIA GPU.
  - Ollama running locally or on a dedicated server.
  - Qdrant in `:memory:` or persistent mode (e.g., Docker volume).

- **Steps**:
  1. Clone repo and install dependencies.
  2. Configure environment:

     ```bash
     cp .env.example .env
     # Edit .env for OLLAMA_BASE_URL, QDRANT_URL
     ```

  3. Run with gunicorn for production:

     ```bash
     pip install gunicorn
     gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8501 streamlit run src/app.py
     ```

  4. Optional: Use nginx as a reverse proxy.

- **Scaling**:
  - Increase workers in gunicorn for high traffic.
  - Use persistent Qdrant storage for large datasets.
  - Optimize VRAM usage with PEFT for large models.

## Best Practices

- Monitor logs: `logs/app.log`.

- Use environment variables for sensitive configs.

- Test GPU support before production deployment.

- Regularly update dependencies via `uv sync`.

For issues, see [../user/troubleshooting.md](../user/troubleshooting.md).
