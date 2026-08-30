"""In-memory metric summaries. Never writes to author result files."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import (
    METRIC_KEYS,
    PERCENT_METRICS,
    ProjectResults,
    load_all_projects,
    load_reference_results,
)


def format_metric(key: str, value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if key in PERCENT_METRICS:
        return f"{value * 100:.{digits}f}%"
    return f"{value:.3f}"


def summarize_runs(project: ProjectResults) -> pd.DataFrame:
    rows = []
    for key in METRIC_KEYS:
        values = [run.metrics[key] for run in project.metric_runs if key in run.metrics]
        if not values:
            rows.append(
                {
                    "metric": key,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n": 0,
                }
            )
            continue
        array = np.array(values, dtype=float)
        rows.append(
            {
                "metric": key,
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "n": int(len(array)),
            }
        )
    return pd.DataFrame(rows)


def runs_as_frame(project: ProjectResults) -> pd.DataFrame:
    rows = []
    for run in project.metric_runs:
        row = {"run": run.run_index + 1, "graph_type": run.graph_type}
        row.update(run.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_confusion(frame: pd.DataFrame) -> dict[str, int]:
    totals = frame[["tp", "tn", "fp", "fn"]].sum()
    return {
        "tp": int(totals["tp"]),
        "tn": int(totals["tn"]),
        "fp": int(totals["fp"]),
        "fn": int(totals["fn"]),
    }


def detection_breakdown(counts: dict[str, int]) -> dict[str, int]:
    tp = counts["tp"]
    fn = counts["fn"]
    fp = counts["fp"]
    return {
        "actual_impacted": tp + fn,
        "detected": tp,
        "missed": fn,
        "predicted_impacted": tp + fp,
        "correct_warnings": tp,
        "false_alarms": fp,
    }


def project_means_table(projects: dict[str, ProjectResults] | None = None) -> pd.DataFrame:
    if projects is None:
        projects = load_all_projects()
    rows = []
    for name, result in projects.items():
        summary = summarize_runs(result)
        if summary["n"].sum() == 0:
            continue
        row = {"project": name}
        for _, item in summary.iterrows():
            row[item["metric"]] = item["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_table(project: ProjectResults) -> pd.DataFrame:
    reference = load_reference_results()
    if reference.empty or "project" not in reference.columns:
        return pd.DataFrame()

    paper = reference[reference["project"].astype(str).str.lower() == project.project.lower()]
    if paper.empty:
        return pd.DataFrame()

    summary = summarize_runs(project)
    means = {row["metric"]: row["mean"] for _, row in summary.iterrows()}
    paper_row = paper.iloc[0]
    rows = []
    for key in METRIC_KEYS:
        if key not in paper_row.index or pd.isna(paper_row[key]):
            continue
        paper_value = float(paper_row[key])
        reproduced = means.get(key, np.nan)
        rows.append(
            {
                "metric": key,
                "paper": paper_value,
                "reproduction": reproduced,
                "absolute_difference": abs(reproduced - paper_value)
                if not np.isnan(reproduced)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def first_hyperparameters(project: ProjectResults) -> dict[str, str]:
    for run in project.metric_runs:
        if run.hyperparameters:
            return run.hyperparameters
    return {}
