# TGN Change Propagation — Visualization Dashboard

A **read-only** Streamlit dashboard for the authors' experiment outputs. It does not train the model, change hyperparameters, or write to `Results/` or `ShuffledData/`.

## Run

From the repository root:

```bash
pip install -r visualization/requirements.txt
streamlit run visualization/app.py
```

The app looks for:

- `Results/Metrics/directed_<project>_results.csv`
- `Results/ConfMatrix/directed_<project>_results_<run>.csv`
- `ShuffledData/<shuffle>/ChangeSets/<project>.csv` (graph reconstruction only)

## Metrics CSV schema

Taken from `TGN_model.py` (the writer), not from the README (which omits columns):

```text
graph_type, project, sensitivity, specificity, ppv, gmean, fmeasure, accuracy, mcc, auc,
lstm_layer_size_1, lstm_layer_size_2, cutoff_value_for_coch, cutoff_change_set_predicted_size, optimizer, epochs
```

Each **row is one experimental shuffle/run**. Metric values on that row are already means over test change-sets.

## Confusion-matrix CSV schema

No header. Each row is one **test change-set**:

```text
tp, tn, fp, fn
```

The dashboard can sum all rows in a run, or show a single change-set.

## Published vs reproduced

Optional. Edit `visualization/reference_results.csv` with values from the paper (0–1 scale, same columns as above). Leave extra rows empty if you do not have published numbers. The dashboard will not invent them.

## Graph view

The authors never save NetworkX graphs, adjacency matrices, or edge lists. `temporal_node_cochanges` lives only in memory during `TGN_model.py`.

The dashboard **reconstructs** directed co-changeability from change-set CSVs with the same formula as `TGN_model.py`:

```text
score(A → B) = |commits(A) ∩ commits(B)| / |commits(A)|
```

in the window starting when both files exist. Interactive graphs use NetworkX + PyVis (arrows, hover labels). This is an input-graph sketch, not an exported TGN snapshot.

If you later want an exact model-state plot, a safe export (without changing training behaviour) would be to write `temporal_node_cochanges[-1]` (or `graph` in `updated_TGN.py`) to JSON/edgelist at the end of a run.
