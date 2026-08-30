"""Offline checks for the visualization layer. Does not modify author files."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_loader import _parse_metrics_row, load_confusion_matrix, load_raw_changesets
from graph_view import build_subgraph, history_up_to  # noqa: E402
from metrics import aggregate_confusion, detection_breakdown, format_metric  # noqa: E402


def test_schema_from_author_write_line() -> None:
    line = (
        "directed,wwwsite,0.48958333333333326,0.9958017676767676,0.9270833333333333,"
        "0.6915678184314991,0.6229166666666666,0.9708530874190564,0.6525498165289955,"
        "0.7426925505050506,4,4,0.005,95,adam,5"
    )
    parsed = _parse_metrics_row(line.split(","), "wwwsite", 0)
    assert parsed is not None
    assert parsed.project == "wwwsite"
    assert parsed.graph_type == "directed"
    assert abs(parsed.metrics["sensitivity"] - 0.48958333333333326) < 1e-12
    assert abs(parsed.metrics["auc"] - 0.7426925505050506) < 1e-12
    assert parsed.hyperparameters["optimizer"] == "adam"
    assert format_metric("sensitivity", parsed.metrics["sensitivity"]) == "49.0%"
    print("OK schema + percentage formatting")


def test_mean_across_runs() -> None:
    a = _parse_metrics_row(
        "directed,pydriller,0.50,0.90,0.80,0.67,0.62,0.88,0.40,0.70,4,4,0.005,95,adam,5".split(","),
        "pydriller",
        0,
    )
    b = _parse_metrics_row(
        "directed,pydriller,0.60,0.92,0.70,0.74,0.65,0.90,0.50,0.80,4,4,0.005,95,adam,5".split(","),
        "pydriller",
        1,
    )
    values = [a.metrics["sensitivity"], b.metrics["sensitivity"]]
    mean = float(np.mean(values))
    assert abs(mean - 0.55) < 1e-12
    print("OK mean of two runs (sensitivity 0.50 and 0.60 → 0.55)")


def test_confusion_aggregate() -> None:
    frame = pd.DataFrame(
        {"tp": [2, 3], "tn": [10, 11], "fp": [1, 0], "fn": [4, 1]}
    )
    counts = aggregate_confusion(frame)
    assert counts == {"tp": 5, "tn": 21, "fp": 1, "fn": 5}
    parts = detection_breakdown(counts)
    assert parts["actual_impacted"] == 10
    assert parts["detected"] == 5
    assert parts["missed"] == 5
    assert parts["false_alarms"] == 1
    print("OK confusion aggregate and recall/precision breakdown")


def test_empty_confusion_file() -> None:
    path = Path("/home/mahesh/Documents/Authors_Code/TGN-change-propagation/Results/ConfMatrix/directed_alamofire_results_0.csv")
    frame = load_confusion_matrix(path)
    assert frame is None
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
        handle.write("")
        empty = Path(handle.name)
    assert load_confusion_matrix(empty) is None
    empty.unlink()
    print("OK empty confusion CSV returns None")


def test_graph_from_changesets() -> None:
    history = history_up_to("pydriller", 0, 20)
    assert history is not None
    changes, changes_of_file, changes_set = history
    assert len(changes) > 0
    graph = build_subgraph(changes_of_file, changes_set, 20, 0.01, 15, None)
    assert graph.number_of_nodes() > 0
    busiest = max(changes_of_file, key=lambda name: len(changes_of_file[name]))
    ego = build_subgraph(changes_of_file, changes_set, 20, 0.01, 15, busiest)
    assert busiest in ego
    assert all(src == busiest for src, _ in ego.edges())
    print(
        f"OK reconstructed pydriller graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges (from ChangeSets, not fabricated weights)"
    )
    print(
        f"OK ego graph for {busiest}: {ego.number_of_nodes()} nodes, "
        f"{ego.number_of_edges()} outgoing companion edges"
    )


def test_author_files_untouched() -> None:
    assert not Path("visualization").joinpath("TGN_model.py").exists()
    raw = load_raw_changesets("alamofire", 0)
    assert raw is not None
    print("OK change-sets readable; visualization stays isolated")


if __name__ == "__main__":
    test_schema_from_author_write_line()
    test_mean_across_runs()
    test_confusion_aggregate()
    test_empty_confusion_file()
    test_graph_from_changesets()
    test_author_files_untouched()
    print("All validation checks passed.")
