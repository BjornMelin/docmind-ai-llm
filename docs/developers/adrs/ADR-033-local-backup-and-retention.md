---
ADR: 033
Title: Local Backup & Retention
Status: Proposed
Version: 1.0
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 030, 031
Tags: backup, retention, local
References:
- shutil, tarfile
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

## Consequences

### Positive Outcomes

- Predictable backups; minimal code

### Dependencies

- Python stdlib

## Changelog

- 1.0 (2025‑09‑02): Proposed manual backup
