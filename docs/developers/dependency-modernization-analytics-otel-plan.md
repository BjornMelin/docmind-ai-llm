# Analytics and Observability Dependency Modernization Plan

Date: 2026-05-01

Status: Implemented in the dependency modernization working tree.

## Objective

Upgrade the highest-leverage, resolver-compatible dependency group that improves
DocMind with minimal architectural churn:

- Plotly 6 native dataframe support for PyArrow-backed analytics.
- PyArrow 24 for current columnar interchange and Parquet compatibility.
- OpenTelemetry 1.41 plus current LlamaIndex OpenTelemetry integration.

This wave deliberately does not broaden into ingestion, RAG, LangGraph,
Qdrant, pandas 3, or DuckDB 1.5. Those lanes have larger behavior and
persistence risk and should remain separate review units.

## Selected Upgrade Group

| Package | Current lock | Target | Decision |
| --- | ---: | ---: | --- |
| `plotly` | `5.24.1` | `6.7.0` | Upgrade |
| `pyarrow` | `21.0.0` | `24.0.0` | Upgrade |
| `opentelemetry-api` | `1.39.1` | `1.41.1` | Transitive/direct lock update |
| `opentelemetry-sdk` | `1.39.1` | `1.41.1` | Upgrade direct floor |
| `opentelemetry-exporter-otlp-proto-http` | `1.39.1` | `1.41.1` | Upgrade direct floor |
| `opentelemetry-exporter-otlp-proto-grpc` | `1.39.1` | `1.41.1` | Upgrade direct floor |
| `opentelemetry-semantic-conventions` | `0.60b1` | `0.62b1` | Resolver update |
| `opentelemetry-proto` | `1.39.1` | `1.41.1` | Resolver update |
| `llama-index-observability-otel` | `0.2.1` | `0.6.0` | Upgrade optional extra |

Resolver dry-run result:

```text
Resolved 335 packages
Updated llama-index-observability-otel v0.2.1 -> v0.6.0
Updated opentelemetry-api/sdk/exporters/proto v1.39.1 -> v1.41.1
Updated opentelemetry-semantic-conventions v0.60b1 -> v0.62b1
Updated plotly v5.24.1 -> v6.7.0
Updated pyarrow v21.0.0 -> v24.0.0
```

Important resolver nuance: the same dry-run reported
`llama-index-instrumentation v0.5.0 -> v0.4.3`. Treat that as a verification
risk for the observability lane. If runtime tests or import probes show
instrumentation regression, keep the core `opentelemetry-*` updates and defer
`llama-index-observability-otel` to a separate LlamaIndex observability PR.

## Hard Cuts

| Candidate | Score | Decision | Reason |
| --- | ---: | --- | --- |
| Analytics plus OpenTelemetry | 8.1 | Selected | User-selected scope. Shared telemetry/analytics surface. Resolver-compatible. |
| Analytics only | 9.0 | Rejected by scope choice | Best minimal lane, but user selected adjacent OTel coverage. |
| Add ingestion/Unstructured 0.22 | 6.7 | Defer | Parsing and table chunking behavior changes deserve a separate ingestion review. |
| pandas 3 | 3.4 | Defer | Latest `llama-index-readers-file==0.6.0` requires `pandas>=2.0.0,<3`. |
| DuckDB 1.5 | 4.8 | Defer | Latest LlamaIndex DuckDB KV/vector integrations require `duckdb<1.4.0`. |
| Custom DuckDB KV adapter | 6.9 | Reject | Would own cache concurrency, JSON extension setup, async facade, and migrations. |
| SQLite ingestion cache pivot | 6.2 | Reject | Reverses ADR-030 and replaces managed LlamaIndex cache with custom persistence. |

## Evidence

- `uv tree --outdated --frozen --all-groups --depth 1` shows Plotly,
  PyArrow, OTel, DuckDB, pandas, Unstructured, spaCy, Torch, and Ruff as
  outdated while LlamaIndex, LangGraph, Qdrant client, Streamlit, OpenAI, and
  Transformers are already current or within their selected release lines.
- The local resolver accepts Plotly 6.7, PyArrow 24, OTel 1.41, and
  `llama-index-observability-otel` 0.6 together.
- The same resolver rejects pandas 3 because
  `llama-index-readers-file==0.6.0` still caps pandas below 3.
- Installed package metadata for `llama-index-vector-stores-duckdb==0.6.0`
  and `llama-index-storage-kvstore-duckdb==0.3.0` still caps DuckDB below 1.4.
- DuckDB current is 1.5.2, but DuckDB 1.4.x is the current LTS line through
  September 2026. Keeping `duckdb>=1.3.2,<1.4.0` is not stale enough to justify
  taking cache implementation ownership.
- LlamaIndex PR 19106 modernized DuckDB vector store internals with relational
  APIs, shared client setup, delete/get/clear support, and tests, which confirms
  the integration is active but still upstream-owned.
- Plotly 6 uses Narwhals for native pandas, Polars, and PyArrow support and
  improves dataframe handling without forcing Pandas conversion.
- DuckDB Python supports Arrow result conversion through `fetch_arrow_table()`
  on the local query result object and direct Arrow registration/querying,
  which matches the analytics refactor.

Primary external sources:

- DuckDB release calendar: <https://duckdb.org/release_calendar.html>
- DuckDB 1.5.2 release notes: <https://duckdb.org/2026/04/13/announcing-duckdb-152.html>
- DuckDB 1.5.0 release notes: <https://duckdb.org/2026/03/09/announcing-duckdb-150.html>
- Plotly 6 migration guide: <https://plotly.com/python/v6-migration/>
- Plotly universal dataframe support: <https://plotly.com/blog/chart-smarter-not-harder-universal-dataframe-support>
- LlamaIndex ingestion pipeline docs:
  <https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/>
- LlamaIndex KV store docs:
  <https://docs.llamaindex.ai/en/stable/module_guides/storing/kv_stores/>
- LlamaIndex DuckDB vector store PR:
  <https://github.com/run-llama/llama_index/pull/19106>

## Implementation Plan

### 1. Update Dependency Bounds

Edit `pyproject.toml`:

```toml
"pyarrow>=24.0.0,<25.0.0"
"plotly>=6.7.0,<7.0.0"
"opentelemetry-sdk>=1.41.1,<2.0.0"
"opentelemetry-exporter-otlp-proto-http>=1.41.1,<2.0.0"
"opentelemetry-exporter-otlp-proto-grpc>=1.41.1,<2.0.0"

observability = [
    "llama-index-observability-otel>=0.6.0,<0.7.0",
]
```

Do not change these in this wave:

```toml
"duckdb>=1.3.2,<1.4.0"
"pandas>=2.2,<3.0"
"llama-index-readers-file>=0.6.0,<0.7.0"
"llama-index-vector-stores-duckdb>=0.6.0,<0.7.0"
"llama-index-storage-kvstore-duckdb>=0.3.0,<0.4.0"
```

Run:

```bash
uv lock \
  --upgrade-package plotly \
  --upgrade-package pyarrow \
  --upgrade-package opentelemetry-sdk \
  --upgrade-package opentelemetry-exporter-otlp-proto-http \
  --upgrade-package opentelemetry-exporter-otlp-proto-grpc \
  --upgrade-package llama-index-observability-otel
```

### 2. Refactor Analytics to Arrow-First

Target file: `src/pages/03_analytics.py`.

Current behavior:

- Imports `pandas as pd`.
- Uses DuckDB `.df()` to materialize Pandas dataframes.
- Passes Pandas dataframes to Plotly Express.

Target behavior:

- Remove the direct Pandas import from the Analytics page.
- Import `pyarrow as pa`.
- Fetch DuckDB query results as PyArrow tables using the local DuckDB Python API
  verified in tests: `con.execute(...).fetch_arrow_table()`.
- Build route-count telemetry data with `pa.table(...)`.
- Pass PyArrow tables directly to `plotly.express` functions.
- Preserve the current connection-close guarantee on every error path.
- Preserve the existing security behavior: sanitized errors only, no raw
  telemetry payload logging.

Expected code shape:

```python
def _query_arrow(con: duckdb.DuckDBPyConnection, sql: str) -> pa.Table:
    return con.execute(sql).fetch_arrow_table()
```

### 3. Keep Pandas as an Eval/Reader Dependency

Do not remove `pandas` from `pyproject.toml` in this wave. It is still required
by current LlamaIndex reader metadata and the RAGAS eval helper surface:

- `tools/eval/run_ragas.py`
- `tests/integration/eval_cli_helpers.py`
- `llama-index-readers-file`

The Analytics page can stop importing Pandas without turning pandas removal into
the objective. Full pandas removal is a later eval/reader decoupling lane.

### 4. Update Tests

Target file: `tests/unit/pages/test_analytics_telemetry_parsing.py`.

Required test changes:

- Replace `pandas as pd` with `pyarrow as pa`.
- Update fake DuckDB query result objects to expose the same Arrow method used
  by `src/pages/03_analytics.py`.
- Keep the existing tests for connection close on success and failure.
- Add or adjust an assertion that `_load_query_metrics()` returns PyArrow
  tables.
- Keep telemetry JSONL route-count parsing deterministic and local-only.

Recommended focused commands:

```bash
uv run pytest tests/unit/pages/test_analytics_telemetry_parsing.py -vv
uv run pytest tests/integration/test_pages_smoke.py -vv
```

### 5. Verify Observability Imports

Add focused import/runtime probes before broad tests:

```bash
uv run python - <<'PY'
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
import llama_index.observability.otel

trace.set_tracer_provider(TracerProvider())
print("otel import probe ok")
PY
```

Then run the existing observability-related tests:

```bash
uv run pytest tests/unit/telemetry -vv
uv run pytest tests/unit/test_observability.py -vv
```

If `tests/unit/test_observability.py` does not exist in the current checkout,
replace it with the actual observability test path discovered by `rg`.

### 6. Update Docs

Update these files only if execution changes the described behavior:

- `docs/developers/adrs/ADR-053-analytics-page-hardening.md`
- `docs/specs/spec-034-analytics-page-hardening.md`
- `docs/specs/traceability.md` if acceptance evidence changes
- `docs/developers/worklogs/CONTEXT.md`

ADR-053 currently documents the earlier Pandas import hardening. Amend it to
say Analytics now uses DuckDB to PyArrow tables and Plotly 6 native dataframe
support, while preserving the original lifecycle and telemetry safety goals.

## Verification Matrix

Run in this order:

```bash
uv lock --check
uv sync --frozen --group test --extra observability --extra eval --group quality
uv run ruff format .
uv run ruff check . --fix
uv run pyright --threads 4 src/pages/03_analytics.py tests/unit/pages/test_analytics_telemetry_parsing.py
uv run pytest tests/unit/pages/test_analytics_telemetry_parsing.py -vv
uv run pytest tests/unit/telemetry -vv
uv run pytest tests/integration/test_pages_smoke.py -vv
uv run pyright --threads 4
uv run python scripts/run_tests.py --fast
```

Final gate for shipping:

```bash
uv run ruff format . && uv run ruff check . --fix && uv run pyright --threads 4 && uv run python scripts/run_tests.py
```

## Stop Rules

Stop and split the PR if any of these occur:

- `llama-index-observability-otel==0.6.0` breaks imports or downgrades
  instrumentation in a way that conflicts with current LlamaIndex runtime.
- Plotly 6 cannot consume the Arrow tables emitted by the local DuckDB API.
- PyArrow 24 breaks GraphRAG Parquet export tests.
- The change set starts touching ingestion, readers, RAG routing, LangGraph
  supervisor code, Qdrant collection logic, or cache persistence.

## Follow-Up Lanes

1. Ingestion parsing modernization:
   Evaluate `unstructured>=0.22,<0.23` for table chunking, formula markdown,
   `unstructured doctor`, PDF render memory safety, and security updates.
2. DuckDB integration watch:
   Track LlamaIndex DuckDB KV/vector packages for a future cap lift to
   DuckDB 1.4 LTS or 1.5 current. Prefer upstream package support over a local
   adapter.
3. Eval/reader dataframe decoupling:
   Remove direct Pandas usage from local eval helpers only after LlamaIndex
   reader metadata permits pandas 3 or the repo replaces that reader path.
4. Tooling lane:
   Evaluate Ruff 0.15 separately because formatter/linter behavior changes can
   churn unrelated files.
