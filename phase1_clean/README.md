# Phase 1 clean temporal-attention implementation

This folder is a small reimplementation of the Phase-1 novelty. It leaves `updated_TGN.py` untouched.

## Main difference

- Existing baseline: train a fresh LSTM for each candidate pair during DFS prediction.
- Clean novelty: train one shared temporal-attention scorer once, freeze it, and reuse it for every candidate pair.

It also adds causal historical features, real elapsed-time recency, and up to 20 previous positive source interactions as attention history.

The directed graph, seed rule, DFS order, mu filtering, rho size cap, and TP/TN/FP/FN accounting are kept compatible with the Germanos reproduction.

## Install

```bash
pip install -r phase1_clean/requirements.txt
```

## Quick smoke test

```bash
python phase1_clean/run.py --project pydriller --shuffle 0 --max-commits 300 --max-epochs 8 --threshold-step 0.05
```

## Full runs

```bash
python phase1_clean/run.py --project alamofire --shuffle 0 --device cpu
python phase1_clean/run.py --project pydriller --shuffle 0 --device cpu
python phase1_clean/run.py --project ant --shuffle 0 --device cpu
```

## Tests

```bash
pytest -q phase1_clean/tests
```

## Outputs

Each run creates `Phase1CleanResults/<project>_shuffle_<n>/` with:

- `summary.json`
- `threshold_grid.csv`
- `test_commit_predictions.csv`
- `candidate_trace.csv`
- `model.pt`

The trace and commit-level outputs are intended to provide visible evidence for why precision, recall, F1, or MCC improved or worsened.

This is a clean reimplementation of the same Phase-1 research idea, not a byte-for-byte reconstruction of the earlier Cursor workspace. Small numerical differences are expected.


## Supported baseline projects

The clean implementation now supports the ten repositories for which the Phase-1 baseline was established:

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

## Run all ten sequentially

Start with shuffle 0:

```bash
python phase1_clean/run_all.py --shuffle 0 --device cpu --continue-on-error
```

For a quicker first pass:

```bash
python phase1_clean/run_all.py --shuffle 0 --device cpu --max-epochs 8 --threshold-step 0.05 --continue-on-error
```

Each project runs in its own process, so memory is released between projects.
