# Phase 1 visualization

The repository uses one read-only Streamlit dashboard for the Phase-1 results.

It does not train models or modify experiment outputs.

## Run

From the repository root:

```bash
pip install -r visualization/requirements.txt
streamlit run visualization/app.py
```

## Views

The dashboard is intentionally limited to three views.

### 1. Baseline results

Shows the reproduced Germanos metrics for the selected repository, including the confusion matrix.

### 2. Clean attention evidence

Shows the Phase-1 improvement in a compact form:

- baseline vs clean F1 and MCC;
- F1/MCC delta by repository;
- recall, precision, false-positive and prediction-size behaviour;
- robustness across all five shuffles when frozen-shuffle outputs are available.

Detailed commit traces and threshold calibration are hidden under optional expanders.

### 3. Input graph

Reconstructs the historical directed co-change graph from the change-set files.

The graph is a reconstruction of the input co-change relationships, not an exported neural-model state and not the predicted change set.

## Data read by the dashboard

Baseline:

```text
Results/Metrics/
Results/ConfMatrix/
ShuffledData/
```

Clean attention:

```text
Phase1CleanResults/<project>_shuffle_0/
Phase1CleanResults/frozen_shuffles/
```

## Important metric note

The reproduced code labels `(Sensitivity + Specificity) / 2` as AUC.

In the Phase-1 analysis this should be described as **Legacy AUC / Balanced Accuracy**, not as a true ROC-AUC.
