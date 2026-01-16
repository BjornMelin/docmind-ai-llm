# Implementation Prompt — Config Discipline (Remaining Polish)

Implements remaining `ADR-050` + `SPEC-031` (post-85% baseline).

**Read first (repo truth):**

- ADR: `docs/developers/adrs/ADR-050-config-discipline-env-bridges.md`
- SPEC: `docs/specs/spec-031-config-discipline-env-bridges.md`
- Requirements: `docs/specs/requirements.md` (NFR-MAINT-003, NFR-SEC-001)
- RTM: `docs/specs/traceability.md`

## Official docs

- <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
- <https://docs.pydantic.dev/latest/concepts/types/#secret-types>

## Tooling

- `rg &quot;os\\.getenv\\\\(&quot; src/config/settings.py` (confirm purge).
- `pytest tests/unit/config/` (add mappings).
- `zen.secaudit` post-changes.
- `zen.codereview` final.

### Parallel Preflight

- `rg &quot;telemetry_disabled|rotate_bytes|sample&quot; src/config/settings.py`
- `rg &quot;img_aes_key_base64|img_kid|img_delete_plaintext&quot; src/config/settings.py`
- Read: `src/config/settings.py`, `tests/unit/config/`

## IMPLEMENTATION EXECUTOR TEMPLATE

### YOU ARE

Autonomous agent for DocMind AI LLM repo. Implement end-to-end: code/tests/docs. Minimal/library-first.

### FEATURE CONTEXT

**Primary Task:** Finalize config discipline: nest Telemetry/ImageEncryption, purge os.getenv from settings.py, add tests, update docs/RTM.

**Why now:** ~85% done (hashing/telemetry wired); close gaps for full DoD/purity.

**Definition of Done (DoD):**

- Nested `TelemetryConfig`/`ImageEncryptionConfig` in `DocMindSettings`; flat fields migrated/aliased.
- Zero `os.getenv` in `src/config/settings.py` (pure Pydantic fields).
- Unit tests: env→nested settings (telemetry disabled/sample/rotate, image enc, hashing validator).
- ADR-050: Accepted; SPEC-031: Implemented; RTM row for NFR-MAINT-003.
- Gates pass; no new debt.

**In-scope:**

- `src/config/settings.py`
- `tests/unit/config/test_telemetry_image_mappings.py` (new)
- `docs/developers/adrs/ADR-050-*.md`
- `docs/specs/spec-031-*.md`
- `docs/specs/traceability.md`

**Out-of-scope:** Renames, new vars/layers.

### HARD RULES

1. Config: `src/config/settings.py` truth; no os.getenv.
2. Style: `ruff format/check/pyright`.
3. Sec: No secret logs; OTLP off-default.

### STEP-BY-STEP PLAN

1. [ ] Nest configs in settings.py; migrate flat fields.
2. [ ] Purge os.getenv(telemetry vars); use Pydantic.
3. [ ] New tests/unit/config/test_telemetry_image_mappings.py (env overrides, validators).
4. [ ] Docs: ADR Accepted, SPEC Implemented, RTM row.
5. [ ] Gates: `ruff/pyright/pytest`.

Commands:

```bash
uv run ruff format . &amp;&amp; uv run ruff check . --fix &amp;&amp; uv run pyright &amp;&amp; uv run python scripts/run_tests.py
```

### ANTI-PATTERN KILL LIST

1. Retain os.getenv in settings.py.
2. Flat configs (nest per SPEC).
3. No tests/docs updates.

### MCP TOOLS

1. `zen.planner` → plan nests/tests.
2. Context7: pydantic-settings nested env.
3. `zen.secaudit` → secrets/egress.
4. `zen.codereview` → final.

### FINAL CHECKLIST

| Req | Status | Proof |
| ---- | -------- | ------- |
| Packaging | | uv sync clean |
| Formatting | | ruff format |
| Lint | | ruff check |
| Types | | pyright |
| Tests | | pytest config + new |
| Docs | | ADR/SPEC/RTM |
| Security | | no secrets/egress |
| Debt | | zero TODO |
| Perf | | fast load |

**EXECUTE UNTIL COMPLETE.**
