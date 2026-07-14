# Operations guide

**Version**: 3.0
**Scope**: Local development, restricted networks, and container operations

## Install and run

DocMind uses Python 3.12 or 3.13 and `uv`. The default lockfile path is CPU-first and uses PyTorch's official CPU wheel index.

```bash
uv sync --frozen
cp .env.example .env
uv run streamlit run app.py
```

Use the optional NVIDIA CUDA 12.8 dependency set only on a compatible host:

```bash
uv sync --frozen --no-group cpu --extra gpu
```

The GPU extra is optional. External LLM servers such as vLLM have their own hardware and deployment requirements.

## Prepare models for restricted networks

Fetch the required retrieval and parser models into their canonical caches:

```bash
uv run python tools/models/pull.py \
  --all \
  --cache_dir ./models_cache \
  --parser-defaults \
  --parser-cache-dir ./cache/models
uv run python scripts/parser_health.py --check
```

`--all` stores four pinned snapshots in one Hugging Face cache: BGE-M3, BM42, the BGE reranker, and SigLIP. The runtime resolves that cache from `DOCMIND_EMBEDDING__CACHE_FOLDER`.

The parser health command hashes every Docling layout file against the canonical source manifest and reports mismatches by relative path in `docling.model_issues`. RapidOCR owns and validates the default ONNX models packaged in its locked wheel; the image and test gates prove that initialization and inference succeed with networking disabled.

After the artifacts are present, set the upstream library offline flags when the deployment requires them:

```env
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

These flags control supported Hugging Face clients. They do not prove that the host has no other network path. DocMind separately rejects non-loopback service endpoints unless `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true` and the endpoint is allowed by policy.

## Operate the parser

DocMind has one parser path: Docling conversion, pypdfium2 PDF inspection and rasterization, and RapidOCR CPU OCR. There is no parser backend selector or remote parser fallback.

Searchable-PDF export is an optional artifact step:

```bash
uv sync --frozen --extra searchable-pdf
```

OCRmyPDF export is fail-open and runs only on POSIX systems. Windows users can run it through WSL2. Timeout, cancellation, and failure handling terminate and reap the complete subprocess group. A searchable-PDF export failure does not change the parser result.

See [Parser runtime validation](parser-runtime-validation.md) for the model preflight and reproducible benchmark command.

## Operate Qdrant safely

### Start a local instance

Use the repository launcher for local Qdrant:

```bash
./scripts/start_qdrant_local.sh
```

The launcher publishes REST and gRPC exactly on `127.0.0.1:6333` and `127.0.0.1:6334`. If a container with the selected name already exists, the launcher reuses it only when both bindings match exactly. It refuses missing, wildcard, IPv6, remapped, or otherwise different bindings so an operator can inspect and resolve the container without automatic deletion.

The launcher accepts these optional controls:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCMIND_QDRANT_CONTAINER` | `docmind-qdrant-local` | Container name |
| `DOCMIND_QDRANT_IMAGE` | `qdrant/qdrant:v1.18.2` | Container image |
| `DOCMIND_QDRANT_PORT` | `6333` | Loopback REST port |
| `DOCMIND_QDRANT_GRPC_PORT` | `6334` | Loopback gRPC port |
| `DOCMIND_QDRANT_STORAGE` | `./data/qdrant-local` | Host storage path |

### Check collection compatibility

Inspect the configured named-vector collection without mutation:

```bash
uv run python scripts/qdrant_schema.py check
```

Application startup creates a missing collection. It accepts a compatible collection and refuses an incompatible one. Runtime code does not mutate or replace an incompatible schema.

### Rebuild a proven-empty collection

Stop every process that can write to the collection before running:

```bash
uv run python scripts/qdrant_schema.py rebuild-empty
```

`rebuild-empty` reads the exact point count, rebuilds only when that count is zero, and refuses non-empty collections or count failures with exit status 2. Qdrant does not atomically lock the count-and-delete sequence, so writer quiescence is an operator requirement. The command never treats a count error as an empty collection.

Do not delete a collection to resolve a vector mismatch. Preserve or migrate non-empty data, or configure a new collection name.

## Run containers

The production compose overlay runs the application with a read-only root filesystem and writable mounts for required data. The Docker build uses the CPU dependency group unless the build is intentionally configured otherwise.

Container health has one implementation owner:

```bash
python scripts/container_health.py
```

The Dockerfile and `docker-compose.prod.yml` use this script as the recurring
liveness probe. It only opens a TCP connection to the local Streamlit port and
does not make an HTTP request. The entrypoint owns the one-time parser dependency
and Docling-model readiness check before Streamlit starts.

## Choose an LLM service

DocMind supports local or policy-approved OpenAI-compatible services. The application does not bundle an inference server.

| Backend | Connection model | Operational note |
| --- | --- | --- |
| Ollama | Native client | Default local backend; DocMind pins the Python client to 0.6.2 |
| vLLM | OpenAI-compatible HTTP | Run and size the server separately |
| llama.cpp | OpenAI-compatible HTTP | Run the server separately |
| LM Studio | OpenAI-compatible HTTP | Run the desktop service separately |

Keep `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false` for loopback-only deployments. Enabling a remote service is a deliberate policy change and may also require an endpoint allowlist entry.

## Tune retrieval and agents

SigLIP is the only image embedding and visual reranking backend.

The agent decision timeout is measured in seconds:

```env
DOCMIND_AGENTS__DECISION_TIMEOUT=200
```

The accepted range is `10s` to `1000s`. Retrieval reranker settings ending in `_MS` remain milliseconds and have separate budgets.

## Protect state

### Reclaim orphaned Qdrant generations

Snapshot activation deliberately leaves old physical collections intact because
online readers do not hold collection leases. To inspect reclaimable generations:

1. Stop every DocMind application process, background worker, and maintenance
   command. Keep Qdrant reachable.
2. Run `uv run python scripts/cleanup_collections.py --confirm-app-stopped`.
   This is a dry run and does not mutate Qdrant.
3. Review `orphan_candidates`. The command fails closed if `CURRENT`, any
   canonical retained snapshot, or `data/.deployment-id` cannot be verified.
   It ignores collections owned by another deployment UUID.
4. Run the same command with `--delete` only after the dry-run set is accepted.
   Re-run the dry run and require an empty candidate list.

Do not run this command while any reader or writer is live. The canonical writer
lock excludes snapshot writers, but it cannot prove that a reader has released an
older immutable collection.

### Preserve activation identity and journals

`data/.deployment-id` binds retained snapshots to their DocMind-owned Qdrant
collections. DocMind creates this canonical UUID file atomically for a new data
directory. If the file is missing or invalid after `CURRENT` or a canonical
snapshot exists, startup fails closed instead of assigning a new owner. Preserve
the file with the entire data directory and every recovery point.

DocMind resolves source and snapshot crash journals while it holds the snapshot
writer lock. Promotion journals under `data/.upload-transactions/` either finish
the exact generation named by `CURRENT` or roll promoted files back to pending
storage. Quarantine journals under `data/.quarantine/` either restore a source
before commit or discard it after the same physical collection generation commits.
`data/storage/.activation-transaction.json` removes an uncommitted promoted
snapshot or retires itself after `CURRENT` names the verified destination. Unknown,
changed, or unsafe journal contents fail closed for operator review.

### Create a backup

Stop DocMind, quiesce every data writer, and keep the single-node Qdrant service
reachable. Then create the backup:

```bash
DOCMIND_BACKUP_ENABLED=true uv run python scripts/backup.py create --json
```

The default command captures authoritative uploads, `data/.deployment-id`, the
active snapshot, exact Qdrant collections, cache, chat database, and existing
content-addressed artifacts. Add `--include-analytics`, `--include-logs`, or
`--include-env` only when those files belong in the recovery point. Review
`.env` separately because it contains credentials.

DuckDB permits an online native copy only inside the process that owns its
writer. The command captures the ingestion cache and optional analytics
database through `COPY FROM DATABASE`. It captures
`settings.chat.sqlite_path` through SQLite's online backup API. These native
copies include committed write-ahead log (WAL) state. A manual copy of only a
database's main file can omit committed data.

The built-in Qdrant capture is intentionally single-node. It marks a run
incomplete rather than presenting one node's snapshot as a complete distributed
backup. Use Qdrant's per-node procedure for a distributed deployment.

Inspect `manifest.json` after every run. A complete recovery point must satisfy
all of these conditions:

- `complete` is `true` and `warnings` is empty
- `included` contains `cache_db`, `snapshots`, `chat_db`, `uploads`,
  `deployment_identity`, and `qdrant_snapshots`
- `activation.snapshot` names the copied `CURRENT` snapshot
- `activation.deployment_id` matches copied `data/.deployment-id`
- `databases.cache_db` and `databases.chat_db` bind each required database to
  its exact backup-relative path, byte count, and SHA-256 digest
- `uploads.files` inventories every copied upload and matches the active corpus
- `artifacts.files` inventories every copied artifact when `artifacts` is listed
- Qdrant `version` is present and its collection records match both names in
  `activation.collections`
- every Qdrant record has `filename`, `size_bytes`, `checksum`, and exact
  `point_count`

`maintenance_warnings` reports cleanup debt after recovery data was verified.
It does not make a backup incomplete. Resolve it before the next maintenance
window, but do not discard the recovery point.

The `create` command exits with status 0 after either a complete recovery point
or an incomplete diagnostic capture. Automation must read `manifest.json` and
require `complete=true` with an empty `warnings` list before treating the result
as recoverable. Do not use process status as the completeness signal.

Use `--diagnostic-no-uploads` or `--no-qdrant-snapshot` only to collect
diagnostic state. Either flag produces an `incomplete-backup_<timestamp>_<id>`
directory. Any recoverability warning has the same result. Diagnostic captures
never prune older recovery points. Complete directories use
`backup_<timestamp>_<id>` and enter retention only after their data is
revalidated.

### Restore a backup

Restore is intentionally an operator-run procedure rather than a CLI command.
It replaces live application state, so first stop DocMind and quiesce every
Qdrant writer. Provision a fresh Qdrant target and keep the prior instance plus
the current files until the restored application passes the verification steps.

1. Open the selected backup's `manifest.json` and apply every completeness check
   from the previous section. Record `app_version`, `activation.snapshot`,
   `activation.deployment_id`, both `activation.collections` values, and each
   Qdrant source version, collection, checksum, and exact `point_count`. Do not
   restore an `incomplete-backup_*` directory.
2. Move each existing target and its sidecars into a separate rollback
   directory. For SQLite, preserve the main file plus any `-wal` and `-shm`
   files. For DuckDB, preserve the main file plus any `.wal` file. Preserve the
   current `data/storage/` tree as one unit.
3. Copy only artifacts listed in `included`, using these mappings:

   | Backup artifact | Restore target |
   | --- | --- |
   | `cache/<database name>` | `settings.cache.ingestion_db_path` |
   | `data/<chat database name>` | `settings.chat.sqlite_path` |
   | `data/analytics/<database name>` | the configured `settings.analytics_db_path` |
   | `data/storage/` | `settings.data_dir/storage/` |
   | `data/.deployment-id` | `settings.data_dir/.deployment-id` |
   | `data/artifacts/` | `settings.artifacts.dir` or `settings.data_dir/artifacts/` |
   | `data/uploads/` | `settings.data_dir/uploads/` |

   Copy the native backup database itself, not source-side WAL files. Restore
   `.env` only after reviewing it field by field; never overwrite current
   credentials or endpoint policy blindly.
4. Start the fresh Qdrant target and keep it writer-quiesced. Do not upload into
   the prior instance: the restore must use the exact physical collection names,
   so the two generations cannot coexist there. The fresh target must share the
   source major and minor version, and its patch version must be at least the
   source patch (for example, `1.4.1` may restore to `1.4.1` or newer `1.4.x`).
   Restore every file under `qdrant/<collection>/<snapshot name>` to the exact
   physical name recorded in `activation.collections`. The restored filesystem
   manifest references these names directly; configured base collection names
   cannot override them. Collection aliases are not included. With `QDRANT_URL`,
   `TARGET_COLLECTION`, `SNAPSHOT`, and the manifest's optional `CHECKSUM` set:

   ```bash
   upload_url="$QDRANT_URL/collections/$TARGET_COLLECTION/snapshots/upload?wait=true&priority=snapshot"
   if [[ -n "${CHECKSUM:-}" ]]; then
     upload_url="$upload_url&checksum=$CHECKSUM"
   fi
   curl --fail-with-body -X POST \
     "$upload_url" \
     -H "api-key: $QDRANT_API_KEY" \
     -F "snapshot=@$SNAPSHOT"
   ```

   Omit the `api-key` header for an unauthenticated local Qdrant instance. Verify
   the restored collection's exact count equals the manifest `point_count`.
5. Before startup, confirm restored `data/.deployment-id` still equals
   `activation.deployment_id`. Start DocMind. Open an existing session and chat,
   query the restored ingestion-cache DuckDB and, when included, the analytics
   DuckDB, then exercise text and image retrieval against the restored physical
   collections. Keep the filesystem rollback directory and prior Qdrant instance
   until ingestion, retrieval, artifact rendering, chat history,
   snapshot loading, and a relationship query through the GraphRAG tool pass a
   real smoke test after a process restart.

If any check fails, stop DocMind, restore the preserved filesystem rollback set
as one unit, and switch back to the preserved Qdrant instance before restarting.
Rollback does not depend on reversing a destructive upload. Retire the prior
instance only after accepting the restored state.

Production deployments must replace `DOCMIND_HASHING__HMAC_SECRET` with a unique secret of at least 32 bytes. When `DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=true`, set `DOCMIND_IMG_AES_KEY_BASE64` to a base64-encoded 32-byte key and manage it outside source control.

## Verify a release candidate

Run the repository's standard quality gates before deployment:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/unit tests/integration -q --cov=src --cov-branch --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json --cov-fail-under=80 --junitxml=junit.xml
```

Use the locked uv environment or the repository Docker image for supported
installations. DocMind does not publish a Python wheel or promise an installed
library API.

See the [configuration reference](configuration.md) for the complete environment schema.
