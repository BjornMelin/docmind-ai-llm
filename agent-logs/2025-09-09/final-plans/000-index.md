# Title: DocMind AI — Final Plans Index

**Date:** 2025-09-09

This folder collects the comprehensive final planning, research, decisions, and code needed to complete the remaining features of DocMind AI. It consolidates GPT‑5 PRO code, Claude 4 Sonnet plans (005), prior research (004), and our repo audit into an actionable package.

## Documents

- **001-exec-summary.md** — Executive summary and scope
- **002-decisions-adr-specs.md** — Final decisions, ADR/SPEC updates, RTM corrections
- **003-ui-multipage-impl.md** — SPEC‑008: Streamlit multipage UI detailed implementation
- **004-analytics-duckdb-impl.md** — ADR‑032: Local analytics DB + charts
- **005-evaluation-harness-impl.md** — SPEC‑010 + ADR‑039: BEIR + RAGAS harness
- **006-model-cli-impl.md** — SPEC‑013 + ADR‑040: Model pre‑download CLI
- **007-graphrag-impl.md** — SPEC‑006: GraphRAG exports, toggles, counters
- **008-observability-security-impl.md** — SPEC‑012 + ADR‑038/039: telemetry + security posture
- **009-task-checklists.md** — Detailed step‑by‑step tasks and subtasks (per file)
- **010-acceptance-criteria-and-tests.md** — Gherkin AC and test plans
- **011-code-snippets.md** — Canonical code blocks from GPT‑5 PRO and research (ready to implement)
- **012-rtm-updates.md** — Requirements traceability matrix updates
- **013-risk-and-mitigation.md** — Risks, performance budgets, contingencies

## Usage

- Start with 001 and 002 to align on decisions.
- For each feature area, read its implementation doc (003..008), then the checklists in 009, AC/tests in 010, and refer to code in 011.
- Apply RTM and ADR/SPEC changes per 012 and 002.
