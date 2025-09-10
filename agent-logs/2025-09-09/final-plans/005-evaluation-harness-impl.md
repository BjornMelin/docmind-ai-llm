# SPEC-010 + ADR-039 — Offline Evaluation Harness (BEIR + RAGAS)

Date: 2025-09-09

## Purpose

Deliver offline evaluation harnesses for IR (BEIR metrics) and end-to-end RAG (RAGAS), writing results to a leaderboard CSV. Complement existing DeepEval (ADR‑012) used for quick CI gates.

## Prerequisites

- `beir` and `ragas` installed
- Data prepared locally (tiny BEIR dataset, a CSV for RAGAS)
- App models cached (see model CLI doc)

## Files to Create

- `tools/eval/run_beir.py` — BEIR IR metrics CLI
- `tools/eval/run_ragas.py` — RAGAS E2E metrics CLI
- `data/eval/README.md` — dataset instructions and usage examples

Code references: final-plans/011-code-snippets.md (Sections 6 and 7)

## Imports and Libraries

- BEIR: `from beir.datasets.data_loader import GenericDataLoader`, `from beir.retrieval.evaluation import EvaluateRetrieval`, `from beir import util`
- RAGAS: `from ragas import evaluate`, `from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision`
- App: `from src.retrieval.hybrid import ServerHybridRetriever`, agent/coordinator for answers

Example imports:

```python
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from src.retrieval.query_engine import ServerHybridRetriever
```

## BEIR Runner (run_beir.py)

### Responsibilities

- Load BEIR corpus/queries/qrels
- Ensure a Qdrant collection with named vectors (build if missing)
- Use our `ServerHybridRetriever` to produce results in BEIR’s format
- Compute `NDCG@10`, `Recall@k`, and `MRR` with `EvaluateRetrieval`
- Save `.run.trec`, JSON metrics, and append `leaderboard.csv`

### Steps

1) Parse CLI args: `--data_dir`, `--k`, `--results_dir`, optional `--collection`, `--qdrant_url`, `--max_docs`.
2) Load: `corpus, queries, qrels = GenericDataLoader(args.data_dir).load(split="test")`.
3) Ensure Qdrant collection schema (named vectors) and index corpus (payloads include `doc_id`, `text`), respecting `--max_docs` for quick runs. Prefer using our storage helpers (e.g., `src.utils.storage.create_vector_store`) to match collection schema.
4) For each query, call `ServerHybridRetriever.retrieve(qtext)` (method name is `retrieve`) with fused buffer ≥ k, then map NodeWithScore list to `{qid: {doc_id: score}}` using `n.node.metadata["doc_id"]` and `n.score`.
5) Compute metrics and write outputs.

See also: ensure-collection helper in final-plans/011-code-snippets.md (Section 13).

## RAGAS Runner (run_ragas.py)

### Responsibilities

- Load a CSV with `question`, `ground_truth`, optional `contexts` JSON list
- For each question, run the agent (or fallback retrieval for contexts)
- Compute `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`
- Append metrics to `leaderboard.csv`

### Steps

1) Parse CLI args: `--dataset_csv`, `--results_dir`, `--top_k`.
2) Read CSV; normalize contexts (retrieve if missing).
3) For each question, invoke `MultiAgentCoordinator.process_query` to get an answer; if contexts empty, retrieve top‑k texts via `ServerHybridRetriever.retrieve` and extract `node.text`.
4) Evaluate with RAGAS, passing a local evaluator wrapper if required by selected metrics.
5) Append mean scores to `leaderboard.csv` with timestamp.

## Acceptance Criteria

- BEIR CLI runs on a tiny dataset and writes metrics + TREC runfile + `leaderboard.csv`.
- RAGAS CLI runs on a small CSV and appends mean metrics to `leaderboard.csv`.
- Both operate offline once data/models are cached.

Gherkin:

```gherkin
Feature: Offline evaluation harnesses
  Scenario: BEIR IR evaluation
    Given corpus and queries from a tiny BEIR dataset
    When I run run_beir.py with k=10
    Then NDCG@10, Recall@10, MRR are computed and saved
    And leaderboard.csv is appended

  Scenario: RAGAS E2E evaluation
    Given a CSV of questions and ground_truth (and optional contexts)
    When I run run_ragas.py
    Then faithfulness and answer_relevancy are computed with context metrics
    And leaderboard.csv is appended
```

## Testing and Notes

- Provide CLI smoke tests with mocks for BEIR and RAGAS modules to avoid network.
- For BEIR indexing, support `--collection` to reuse an existing collection.
- Document dataset preparation in `data/eval/README.md` and offline env flags.

## How to Run (Examples)

```bash
# BEIR (scifact) example
python tools/eval/run_beir.py --data_dir data/beir/scifact/ --k 10 --results_dir eval/results

# RAGAS example
python tools/eval/run_ragas.py --dataset_csv data/eval/sample.csv --results_dir eval/results
```

## Cross-Links

- Hybrid retrieval usage and server-side fusion: SPEC‑004 (docs/specs/spec-004-hybrid-retrieval.md)
- Code snippets for BEIR/RAGAS and tests: 011-code-snippets.md (Sections 6–8)

## No Backwards Compatibility

- Replace any legacy evaluation scripts with the new CLIs (`tools/eval/run_beir.py`, `tools/eval/run_ragas.py`). Delete older evaluation tools if present and update documentation and test invocations accordingly.
