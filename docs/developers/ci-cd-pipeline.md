# Verify DocMind in continuous integration

DocMind uses separate workflows for application, container, release-contract, and documentation checks. This page maps each hosted gate to its local command.

## Required workflow jobs

The `CI` workflow runs these jobs for pull requests and `main` pushes:

| Job | Required behavior |
| --- | --- |
| `build-test` | CPython 3.12.13, CPU dependency profile, Ruff, Pyright, end-to-end tests, Qdrant system smoke, unit and integration coverage, and schema validation |
| `compatibility-python-313` | CPython 3.13.12 base environment, full unit suite, and focused GraphRAG tests |
| `qdrant-fusion` | RRF and DBSF queries against the pinned Qdrant service |
| `container-static` | Dockerfile, Compose, non-root, image-tag, environment, and entrypoint policy |
| `container-build` | Production image build and canonical liveness smoke |

Generated Release Please pull requests also run `release-contract`, which checks version agreement and the lockfile. The documentation workflow skips generated release pull requests because the source change was validated before merge.

The `Documentation Quality Gates` workflow runs link validation, structural parity, and Markdownlint for other pull requests and `main` pushes.

## Reproduce application checks

Run the non-mutating quality checks locally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ruff check src --config ruff-core.toml
uv run pyright --threads 4
uv run pytest tests/unit tests/integration -q --no-cov
```

Run the opt-in acceptance suites when your change affects those paths:

```bash
DOCMIND_RUN_E2E=1 uv run pytest tests/e2e -q --no-cov
DOCMIND_RUN_SYSTEM=1 \
  DOCMIND_QDRANT_SYSTEM_URL=http://127.0.0.1:6333 \
  uv run pytest tests/system/test_e2e_offline.py -q --no-cov
```

Continuous integration sets Hugging Face and Transformers offline flags. Those flags prevent model downloads after prefetch; they do not prove zero network egress.

## Reproduce documentation checks

Run the same documentation commands used by the hosted workflow:

```bash
uv run python scripts/check_links.py
uv run python scripts/verify_structural_parity.py
npx --yes markdownlint-cli@0.47.0 \
  --disable MD013 MD033 MD041 -- 'docs/**/*.md'
```

Schema validation runs in the application workflow:

```bash
uv run python scripts/validate_schemas.py
```

## Diagnose Streamlit AppTest timeouts

Reproduce an application-test timeout with the focused user-interface suite:

```bash
uv run pytest tests/integration/ui -vv --no-cov
uv run pytest tests/integration/ui \
  --durations=20 --durations-min=0.5 --no-cov
CI=1 TEST_TIMEOUT=40 \
  uv run pytest tests/integration/ui -q --no-cov
```

`tests/helpers/apptest_utils.py` owns AppTest timeout selection. Keep heavy services stubbed and assert user-visible behavior instead of brittle rendered strings.
