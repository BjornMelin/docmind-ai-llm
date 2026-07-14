# Testing patterns

DocMind uses pytest as its single test runner. Tests should prove behavior at
the narrowest real boundary that carries the contract; avoid tests that only
instantiate unused models, mirror implementation details, or turn missing
dependencies into a pass.

## Test tiers

- Unit tests are deterministic, offline, and do not require external services.
- Integration tests combine real repository components with lightweight local
  boundaries.
- System tests opt in to an explicitly provisioned service such as Qdrant.
- E2E tests exercise supported Streamlit and workflow behavior.
- GPU tests are manual and require the locked GPU profile.

Run the native Pytest lanes:

```bash
uv run pytest tests/unit -q --no-cov
uv run pytest tests/integration -q --no-cov
uv run pytest tests/unit tests/integration -q --no-cov
uv run --no-sync pytest -m requires_gpu --no-cov
```

Run coverage with every required output and the 80 percent floor:

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

## Boundary-first rules

1. Patch network, filesystem, clock, process, or model-loading boundaries—not
   the function whose decisions the test is meant to prove.
2. Prefer `tmp_path`, real SQLite/DuckDB files, and checked-in service topology
   over hand-built storage mocks.
3. Use `responses` for synchronous HTTP boundaries. Unit tests also have an
   autouse network guard.
4. Run real service integrations through CI service containers or the
   repository Compose topology so lifecycle and health failures remain visible.
5. Stub model/library objects at their public boundary; never download models
   or make network requests in the unit suite.
6. Assert observable outputs, persisted state, and failure semantics. Avoid
   assertions about private call order unless that ordering is the contract.

Example filesystem boundary:

```python
def test_manifest_round_trip(tmp_path: Path) -> None:
    manager = SnapshotManager(tmp_path)
    workspace = manager.begin_snapshot()
    manager.write_manifest(
        workspace,
        index_id="test",
        graph_store_type="property_graph",
        vector_store_type="qdrant",
        corpus_hash="sha256:corpus",
        config_hash="sha256:config",
        versions={"app": "test"},
    )

    snapshot = manager.finalize_snapshot(workspace)

    assert snapshot.is_dir()
    assert (snapshot / "manifest.jsonl").is_file()
```

## Parser tests

The parser boundary must cover successful Docling conversion, OCR result
normalization, physical page mapping, limits, artifact integrity failures, and
worker timeout/cancellation. Stub the external parser engines and model files;
do not bypass `parse_document_sync` or its canonical backend contracts.

```bash
uv run pytest tests/unit/processing/test_parser_contract.py \
  --cov=src.processing.parsing \
  --cov-branch \
  --cov-report=term-missing
```

For runtime quality and performance, use the isolated schema-3 harness against a
clean commit:

```bash
uv run python scripts/benchmark_parsing.py \
  --generate-minimal-fixtures \
  --output cache/benchmarks/parsing/results.json
```

Compare benchmark artifacts only across equivalent hardware, model caches,
fixtures, and commands. A missing baseline, missing sample, failed child
process, or incomparable runtime is not a successful regression result.

## Persistence tests

Persistence tests should exercise successful writes and reloads as well as
cleanup on failure. For snapshot/corpus behavior, cover added, removed,
renamed, modified, and symlinked source paths; assert that stale cache entries
cannot survive a corpus identity change.

Use real temporary SQLite connections and close them in the fixture that owns
their lifecycle. Do not share a connection across threads.

## Safe logging tests

Use `src.utils.log_safety` for exception fingerprinting and redaction. Tests may
capture Loguru output and JSONL events, but must assert that canary secrets and
raw input never appear in either channel.

```python
monkeypatch.setattr(
    "src.utils.log_safety.log_jsonl",
    lambda event: captured_events.append(dict(event)),
)
log_error_with_context(RuntimeError(canary), operation="test")
```

## Review checklist

- The test fails when the behavior regresses.
- No missing package, service, artifact, or baseline is interpreted as success.
- Assertions target the supported runtime contract.
- Fixtures clean up files, processes, threads, connections, and global state.
- Unit tests perform no network egress or model download.
- Focused coverage protects the changed owner even when global coverage passes.
