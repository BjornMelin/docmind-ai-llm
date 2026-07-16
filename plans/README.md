# DocMind UI Modernization Plans

Generated from a read-only `improve` audit on 2026-07-16 and rebased on commit
`9accab1`. Execute in order. Each executor must read its plan completely,
honor STOP conditions, and update the status row when work is complete.

The target is a polished local-first Streamlit application, not a framework
rewrite. Native Streamlit and Python standard-library capabilities are the
default. Preserve every current Chat, Documents, Analytics, Settings,
snapshot, privacy, and fail-closed behavior unless a plan explicitly changes
the visible contract.

## Decision framework

Scores use the repository-wide weights: solution leverage 35%, application
value 30%, maintenance/cognitive load 25%, architectural adaptability 10%.

| Plan | Solution leverage | Application value | Maintenance | Adaptability | Weighted score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 001 Mutation and job-ownership safety | 9.4 | 9.8 | 9.1 | 9.5 | **9.5** |
| 002 Truthful Chat states | 9.5 | 9.2 | 9.4 | 9.3 | **9.4** |
| 003 Native page-startup performance | 9.2 | 8.8 | 9.5 | 9.5 | **9.2** |
| 004 Rendered-browser acceptance lane | 9.0 | 8.7 | 9.1 | 9.4 | **9.0** |
| 005 Guided workspace shell | 9.0 | 9.4 | 9.1 | 9.3 | **9.2** |
| 006 Documents corpus workspace | 9.1 | 9.6 | 9.0 | 9.1 | **9.2** |
| 007 Chat evidence presentation | 8.8 | 9.5 | 8.8 | 9.2 | **9.1** |
| 008 Settings information architecture | 9.0 | 9.3 | 9.1 | 9.2 | **9.1** |
| 009 Analytics operations view | 8.3 | 8.5 | 8.8 | 8.9 | **8.5** |

## Execution order and status

| Plan | Title | Priority | Effort | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| 001 | Preserve job ownership and serialize corpus mutations | P1 | M | — | TODO |
| 002 | Make Chat loading and generation states truthful | P1 | S–M | 001 | TODO |
| 003 | Remove duplicate and eager page-startup work | P1 | M | 001, 002 | TODO |
| 004 | Add rendered-browser and accessibility acceptance | P1 | M | 001–003 | TODO |
| 005 | Add the guided workspace shell | P2 | M | 001–004 | TODO |
| 006 | Build the Documents corpus workspace | P2 | M | 001, 003, 004, 005 | TODO |
| 007 | Polish Chat and current-answer evidence | P2 | M | 002, 003, 004, 005 | TODO |
| 008 | Reframe Settings without changing lifetimes | P2 | M | 001, 004, 005 | TODO |
| 009 | Turn Analytics into a bounded operations view | P3 | S–M | 004, 005 | TODO |

Status values: `TODO`, `IN PROGRESS`, `DONE`, `BLOCKED: <reason>`, or
`REJECTED: <reason>`.

## Dependency notes

- Plan 001 makes long-running work observable and safe before layout changes
  move its controls.
- Plan 003 depends on the process-stable job owner from Plan 001 so cache and
  import work cannot orphan a running task.
- Plan 004 depends on Plan 003's production-safe offline Chat state, then lands
  before visual restructuring so desktop/mobile, focus,
  console, and accessibility regressions have a durable gate.
- Plans 005–009 split the redesign by owned surface. Each changes hierarchy and
  presentation only after its correctness/performance/browser prerequisites
  are proven; no executor owns all pages at once.

## Findings considered and deferred

- FastAPI/React rewrite: rejected. The accepted UI ADR scores native
  programmatic Streamlit at 9.0 and no current capability justifies a second
  frontend stack.
- GSAP, injected JavaScript, or broad custom CSS: rejected. These fight
  Streamlit reruns and DOM ownership; use native status, dialog, fragment,
  container, dataframe, and navigation primitives.
- Celery or another queue: rejected. The local thread manager is sufficient;
  its ownership and exclusivity contracts need correction, not replacement.
- Persisted per-message source evidence: high-value follow-up, but it changes
  checkpoint/persistence contracts and should be specified after Plans 001–005
  prove the redesigned evidence surface. Current source rendering must remain.
- Broad dependency churn: rejected. The lock currently resolves modern
  Streamlit and Plotly releases; no major-version gap was evidenced.

## Audit limits

No live browser or screenshots were available during the audit. Breakpoint
behavior, focus order, contrast, Plotly clipping, touch targets, and motion
quality remain `UNVERIFIED` until Plan 004 runs against the rendered app.
