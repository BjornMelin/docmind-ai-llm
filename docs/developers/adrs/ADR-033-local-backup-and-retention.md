---
ADR: 033
Title: Local Backup & Retention
Status: Proposed
Version: 1.1
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 030, 031
Tags: backup, retention, local
References:
- [Python — shutil](https://docs.python.org/3/library/shutil.html)
- [Python — tarfile](https://docs.python.org/3/library/tarfile.html)
---

## Description

Provide a simple, manual backup command that snapshots key artifacts with small rotation. No daemons; no external services.

## Context

Local users need an easy, predictable backup path.

## Decision Drivers

- KISS; explicit user action; safe restore steps

## Alternatives

- Automated tasks — complexity
- External services — default reject

## Decision

Ship a CLI/script that copies cache DB, Qdrant data, and optional docs into a timestamped dir; prune by count.

### Decision Framework

| Option                     | Simplicity (40%) | Safety (30%) | Coverage (20%) | Effort (10%) | Total | Decision      |
| -------------------------- | ---------------- | ------------ | -------------- | ------------ | ----- | ------------- |
| Manual snapshot (Sel.)     | 10               | 9            | 7              | 9            | 9.1   | ✅ Selected    |
| Scheduled background task  | 6                | 8            | 8              | 6            | 6.9   | Rejected      |
| External backup service    | 4                | 9            | 10             | 4            | 6.7   | Rejected      |

## High-Level Architecture

CLI → snapshot dir → retain N most recent

## Related Requirements

### Functional Requirements

- FR‑1: Archive config/data dirs; prune old archives

### Non-Functional Requirements

- NFR‑1: Local‑only; predictable schedule

### Performance Requirements

- PR‑1: Backup completes within 2 minutes for typical data sizes

### Integration Requirements

- IR‑1: Toggle via settings; path configurable

## Design

### Architecture Overview

- On demand, single command snapshots; simple retention by count

### Implementation Details

- Use `tarfile` for archive and `shutil` for file operations

### Configuration

```env
DOCMIND_BACKUP__ENABLED=true
DOCMIND_BACKUP__RETENTION=5
```

## Consequences

### Positive Outcomes

- Predictable backups; minimal code

### Negative Consequences / Trade-offs

- Disk usage grows until pruning occurs

### Ongoing Maintenance & Considerations

- Periodically test restore process end‑to‑end

### Dependencies

- Python stdlib

## Testing

```python
def test_backup_paths(tmp_path):
    # stub: ensure backup/restore paths resolve under tmp
    pass
```

## Changelog

- 1.1 (2025‑09‑04): Standardized to template; added decision framework
- 1.0 (2025‑09‑02): Proposed manual backup
