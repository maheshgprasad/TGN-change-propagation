# Phase 1 clean temporal-attention implementation

This folder contains the Phase-1 improvement built on top of the reproduced Germanos change-propagation framework.

## What changed

The original reproduced implementation trains a fresh LSTM for each candidate pair during DFS prediction.

The clean implementation instead:

- trains one shared temporal-attention scorer;
- uses causal historical features only;
- includes actual elapsed-time recency;
- keeps up to 20 recent positive source interactions as temporal context;
- freezes the trained model before test evaluation; and
- reuses the same scorer for every candidate pair.

The original directed graph, seed rule, DFS traversal, mu filtering, rho prediction-size cap, and TP/TN/FP/FN accounting are preserved.

## Supported projects

The ten repositories with an established Phase-1 baseline are:

- alamofire
- ant
- cassandra
- laravel
- lucene
- monitorcontrol
- pydriller
- react
- rocketmqclients
- spark

## Install

```bash
pip install -r phase1_clean/requirements.txt
```

## Train and evaluate shuffle 0

Run one project:

```bash
python phase1_clean/run.py --project pydriller --shuffle 0 --device cpu
```

Run all ten sequentially:

```bash
python phase1_clean/run_all.py --shuffle 0 --device cpu --continue-on-error
```

Shuffle 0 is used for training, chronological validation, threshold selection, and the first test evaluation.

## Frozen robustness evaluation on shuffles 1-4

After shuffle 0 is complete, evaluate the same trained model on the four remaining within-commit file orderings:

```bash
python phase1_clean/evaluate_frozen_shuffles.py \
  --device cpu \
  --continue-on-error
```

This does **not** retrain the model and does **not** recalibrate the threshold.

For each project it reuses the shuffle-0:

- trained model weights;
- feature preprocessing;
- selected threshold;
- mu value;
- rho prediction-size cap; and
- temporal-attention configuration.

Only the file ordering inside each commit changes.

This is therefore a robustness test for seed and traversal-order sensitivity, not five independent datasets.

## Outputs

Shuffle-0 outputs are stored under:

```text
Phase1CleanResults/<project>_shuffle_0/
```

Important files:

- `summary.json`
- `threshold_grid.csv`
- `test_commit_predictions.csv`
- `candidate_trace.csv`
- `model.pt`

Frozen shuffle 1-4 results are stored under:

```text
Phase1CleanResults/frozen_shuffles/
```

Important aggregate files:

- `all_shuffles_comparison.csv`
- `project_summary.csv`
- `run_manifest.json`

The project summary reports baseline-vs-clean mean F1/MCC and the number of shuffle-level wins for each repository.

## Visualization

Use the existing Streamlit dashboard:

```bash
streamlit run visualization/app.py
```

The dashboard is intentionally kept small:

1. **Baseline results**
2. **Clean attention evidence**
3. **Input graph**

The clean-attention view shows the main comparison first. Commit-level traces and threshold details are optional expanders rather than primary output.

## Tests

```bash
pytest -q phase1_clean/tests
```

## Interpretation

The clean implementation is a controlled replacement of the candidate-scoring stage, not a replacement of the entire Germanos framework.

The intended Phase-1 question is whether richer causal temporal evidence and one reusable attention model can improve impact prediction while removing repeated neural-network fitting from the inference loop.
