# Evaluation Datasets

This folder contains instructions to run the offline evaluation harnesses.

## BEIR (IR metrics)

Prepare a tiny dataset like `scifact` under `data/beir/scifact/` and then run:

```bash
python tools/eval/run_beir.py --data_dir data/beir/scifact/ --k 10 --results_dir eval/results
```

This computes NDCG@10, Recall@10, and MRR@10 and appends a row to `eval/results/leaderboard.csv`.

## RAGAS (E2E metrics)

Prepare a CSV with columns: `question, ground_truth, contexts` (contexts optional). Then run:

```bash
python tools/eval/run_ragas.py --dataset_csv data/eval/sample.csv --results_dir eval/results
```

This computes `faithfulness`, `answer_relevancy`, `context_recall`, and `context_precision` and appends results to `eval/results/leaderboard.csv`.

## Offline Notes

Download models in advance:

```bash
python tools/models/pull.py --all --cache_dir ~/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```
