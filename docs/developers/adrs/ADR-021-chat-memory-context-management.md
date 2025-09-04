---
ADR: 021
Title: Conversational Memory & Context Management
Status: Accepted
Version: 3.3
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 013, 016, 031
Tags: chat, memory, context
References:
- LlamaIndex ChatMemoryBuffer
---

## Description

Implement persistent chat memory with trimming for 128K context. Use built‑in components (ChatMemoryBuffer, simple store) and Streamlit state for UI.

## Context

Users expect follow‑ups to use prior context and survive restarts. Keep it simple and local.

## Decision Drivers

- Persist conversations; trim aggressively
- Minimal moving parts; testable

## Alternatives

- No memory — poor UX
- Custom memory system — overkill

### Decision Framework

| Option                  | UX (40%) | Simplicity (40%) | Perf (20%) | Total | Decision |
| ----------------------- | -------- | ---------------- | ---------- | ----- | -------- |
| Built‑in memory (Sel.)  | 9        | 9                | 8          | 8.8   | ✅ Sel.  |

## Decision

Use ChatMemoryBuffer + lightweight persistence; trim to maintain 128K budget.

## High-Level Architecture

UI ↔ session_state ↔ memory buffer ↔ persistence (SQLite)

## Related Requirements

### Functional Requirements

- FR‑1: Persist chat history across sessions
- FR‑2: Use prior turns for follow‑ups
- FR‑3: Support multiple conversations (session IDs)

### Non-Functional Requirements

- NFR‑1: Memory operations add <100ms overhead at P95
- NFR‑2: Local‑first; no external services required

### Performance Requirements

- PR‑1: Trim to keep ≤128K total tokens with ~8K buffer

### Integration Requirements

- IR‑1: Expose session IDs via UI state (ADR‑013/016)
- IR‑2: Use LlamaIndex chat store for persistence when enabled

## Design

### Implementation Details

```python
def trim_messages(messages, max_tokens=120_000):
    # keep most recent messages within budget
    return messages
```

### Configuration

```env
DOCMIND_CHAT__PERSIST=true
DOCMIND_CHAT__MAX_TOKENS=120000
```

## Testing

- Verify persistence and trimming behavior on long chats
- Snapshot tests for multi‑session separation

## Consequences

### Positive Outcomes

- Predictable, local memory behavior

### Negative Consequences / Trade-offs

- Requires tuning trim thresholds

### Dependencies

- Python: `llama-index`

## Changelog

- 3.3 (2025‑09‑04): Standardized to template; added FR/NFR/PR/IR and config/tests

- 3.2 (2025‑09‑02): Accepted; 128K‑aligned trimming
