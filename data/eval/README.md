# Evaluation Datasets

This folder contains instructions to run the offline evaluation harnesses.

## BEIR (IR metrics)

Prepare a tiny dataset like `scifact` under `data/beir/scifact/` and then run:

```bash
python tools/eval/run_beir.py \
  --data_dir data/beir/scifact/ \
  --k 10 \
  --sample_count 20 \
  --results_dir eval/results
```

This computes nDCG@K, Recall@K, and MRR@K (with dynamic `@{k}` headers) and appends
a row to `eval/results/leaderboard.csv` with fields: `schema_version, ts, dataset, k,
ndcg@{k}, recall@{k}, mrr@{k}, sample_count`.

## RAGAS (E2E metrics)

Prepare a CSV with columns: `question, ground_truth, contexts` (contexts optional). Then run:

```bash
python tools/eval/run_ragas.py \
  --dataset_csv data/eval/sample.csv \
  --ragas_mode offline \
  --sample_count 20 \
  --results_dir eval/results
```

This computes `faithfulness`, `answer_relevancy`, `context_recall`, and `context_precision`
and appends results to `eval/results/leaderboard.csv` with fields:
`schema_version, ts, dataset, faithfulness, answer_relevancy, context_recall, context_precision, sample_count`.

## Offline Notes

Download models in advance:

```bash
python tools/models/pull.py --all --cache_dir ~/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

Schema validation:

```bash
uv run python scripts/validate_schemas.py  # auto-discovers leaderboard.csv files
```
```
