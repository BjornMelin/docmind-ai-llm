# 004 — Test Plan Diff (Additions/Changes)

Unit
- Snapshot round‑trip: persist_dir + from_persist_dir (new)
- Router factory: path_depth default=1 (new)
- Hashing relpath determinism across path order/OS (new)
- Manifest versions present (new)
- Export label preservation when available (new)

Integration
- Chat autoload from snapshot (AppTest): router present when snapshot is non‑stale (new)
- Router composition remains (vector+graph vs vector‑only) (unchanged)
- Exports JSONL/Parquet conditional (unchanged)

E2E
- Optional: resume from snapshot end‑to‑end (new, can be skipped in default CI)

Acceptance (AC‑FR‑009)
- Router composition + fallback (covered)
- Manifest + staleness badge (covered; add autoload behavior)
- Exports JSONL + Parquet (covered; add relation label preservation)

