# Find and run DocMind tests

This reference maps test directories to their source owners and lists the supported local commands.

## Test tiers

- `unit`: isolated domain behavior
- `integration`: cross-component behavior and Streamlit AppTest
- `e2e`: application workflows
- `system`: opt-in local-service validation

## Unit domains

- `agents`: supervisor, agents, tools, and registry
- `analysis`: analysis services
- `config`: settings and runtime binding
- `core`: shared invariants and dependency contracts
- `eval`: evaluation helpers
- `models`: Pydantic and embedding models
- `nlp`: spaCy services and transforms
- `pages` and `ui`: Streamlit helpers and components
- `persistence`: chat, snapshots, artifacts, and checkpoints
- `processing`: parser and LlamaIndex ingestion
- `prompting`: prompt templates and rendering
- `retrieval`: hybrid search, GraphRAG, reranking, and SigLIP
- `scripts`: benchmark, container, release, and Qdrant commands
- `security`: security boundaries
- `telemetry`: event and trace contracts
- `utils`: storage, images, log safety, and shared helpers

Keep domain fixtures beside their tests. Reserve `tests/conftest.py` and `tests/unit/conftest.py` for cross-domain defaults.

## Run tests

```bash
uv run pytest tests/unit/processing/test_parser_contract.py -vv --no-cov
uv run pytest tests/unit -q --no-cov
uv run pytest tests/integration -q --no-cov
uv run pytest tests/unit tests/integration -q --no-cov
uv run --no-sync pytest -m requires_gpu --no-cov
```

Run the coverage gate with explicit native outputs:

```bash
uv run pytest tests/unit tests/integration -q \
  --cov=src \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=html:htmlcov \
  --cov-report=xml:coverage.xml \
  --cov-report=json:coverage.json \
  --cov-fail-under=80 \
  --junitxml=junit.xml
```

Run system tests explicitly:

```bash
DOCMIND_RUN_SYSTEM=1 uv run pytest tests/system -vv
```

`requires_llama` selects tests that need the real required `llama_index.core` package. There is no `llama` extra or `--extras` test lane.

See `docs/testing/testing-guide.md` for markers, dependency profiles, and contribution rules.
