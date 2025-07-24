FROM python:3.12-slim

RUN apt-get update && apt-get install -y libmupdf-dev libmagic1 && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml .
RUN uv venv .venv
RUN . .venv/bin/activate && uv sync --extra gpu

COPY . .

RUN . .venv/bin/activate && python -m spacy download en_core_web_sm

EXPOSE 8501

CMD [". .venv/bin/activate && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
