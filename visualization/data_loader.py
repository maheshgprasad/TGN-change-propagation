"""Read-only loaders for author result CSVs and change-set files.

Schema is taken from TGN_model.py (the code that writes the files), not from
the README, which omits the project name and hyperparameter columns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "Results"
METRICS_DIR = RESULTS_ROOT / "Metrics"
CONF_DIR = RESULTS_ROOT / "ConfMatrix"
DATA_ROOT = REPO_ROOT / "ShuffledData"
REFERENCE_PATH = Path(__file__).resolve().parent / "reference_results.csv"

METRIC_KEYS = (
    "sensitivity",
    "specificity",
    "ppv",
    "gmean",
    "fmeasure",
    "accuracy",
    "mcc",
    "auc",
)

METRIC_LABELS = {
    "sensitivity": "Sensitivity / Recall",
    "specificity": "Specificity",
    "ppv": "PPV / Precision",
    "gmean": "G-Mean",
    "fmeasure": "F-measure / F1",
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "auc": "AUC",
}

PERCENT_METRICS = {
    "sensitivity",
    "specificity",
    "ppv",
    "gmean",
    "fmeasure",
    "accuracy",
    "auc",
}

HYPERPARAM_COLUMNS = (
    "lstm_layer_size_1",
    "lstm_layer_size_2",
    "cutoff_value_for_coch",
    "cutoff_change_set_predicted_size",
    "optimizer",
    "epochs",
)

DISPLAY_NAMES = {
    "alamofire": "Alamofire",
    "ant": "Ant",
    "cassandra": "Cassandra",
    "cassandrawebsite": "Cassandra Website",
    "flutter": "Flutter",
    "gephi": "Gephi",
    "hbase": "HBase",
    "laravel": "Laravel",
    "lucene": "Lucene",
    "monitorcontrol": "Monitor Control",
    "pydriller": "PyDriller",
    "react": "React",
    "rocketmqclients": "RocketMQ Clients",
    "spark": "Spark",
    "wwwsite": "WWW Site",
}

_METRICS_NAME = re.compile(r"^directed_(.+)_results\.csv$", re.IGNORECASE)
_CONF_NAME = re.compile(r"^directed_(.+)_results_?(\d+)\.csv$", re.IGNORECASE)


def display_name(project: str) -> str:
    return DISPLAY_NAMES.get(project.lower(), project.replace("_", " ").title())


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class MetricRun:
    graph_type: str
    project: str
    run_index: int
    metrics: dict[str, float]
    hyperparameters: dict[str, str] = field(default_factory=dict)


@dataclass
class ProjectResults:
    project: str
    graph_type: str
    metric_runs: list[MetricRun]
    confusion_runs: dict[int, pd.DataFrame]
    source_metrics_file: Path | None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_metric_runs(self) -> int:
        return len(self.metric_runs)

    @property
    def confusion_run_ids(self) -> list[int]:
        return sorted(self.confusion_runs)


def _parse_metrics_row(cells: list[str], project_from_name: str, run_index: int) -> MetricRun | None:
    cells = [c.strip() for c in cells]
    if not cells or all(c == "" for c in cells):
        return None

    graph_type = cells[0] if cells[0] else "directed"
    offset = 1
    project = project_from_name
    if len(cells) > 1 and not _is_float(cells[1]):
        project = cells[1]
        offset = 2

    metric_cells = cells[offset : offset + 8]
    if len(metric_cells) < 8:
        return None

    metrics = {}
    for key, raw in zip(METRIC_KEYS, metric_cells):
        try:
            metrics[key] = float(raw)
        except ValueError:
            return None

    hyper_raw = cells[offset + 8 :]
    hyperparameters: dict[str, str] = {}
    for name, value in zip(HYPERPARAM_COLUMNS, hyper_raw):
        hyperparameters[name] = value

    return MetricRun(
        graph_type=graph_type,
        project=project,
        run_index=run_index,
        metrics=metrics,
        hyperparameters=hyperparameters,
    )


def _read_csv_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def discover_projects() -> list[str]:
    names: set[str] = set()
    if METRICS_DIR.is_dir():
        for path in METRICS_DIR.glob("*.csv"):
            match = _METRICS_NAME.match(path.name)
            if match:
                names.add(match.group(1))
    if CONF_DIR.is_dir():
        for path in CONF_DIR.glob("*.csv"):
            match = _CONF_NAME.match(path.name)
            if match:
                names.add(match.group(1))
    return sorted(names)


def load_confusion_matrix(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(
            path,
            header=None,
            names=["tp", "tn", "fp", "fn"],
            usecols=[0, 1, 2, 3],
        )
    except (pd.errors.EmptyDataError, ValueError):
        return None
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(how="all")
    if numeric.empty:
        return None
    return numeric.fillna(0)


def load_project(project: str) -> ProjectResults:
    warnings: list[str] = []
    metrics_path = METRICS_DIR / f"directed_{project}_results.csv"
    metric_runs: list[MetricRun] = []
    graph_type = "directed"

    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        rows = _read_csv_rows(metrics_path)
        for index, cells in enumerate(rows):
            parsed = _parse_metrics_row(cells, project, index)
            if parsed is None:
                warnings.append(f"Skipped unreadable metrics row {index + 1} in {metrics_path.name}.")
                continue
            metric_runs.append(parsed)
            graph_type = parsed.graph_type
    elif metrics_path.is_file():
        warnings.append(f"{metrics_path.name} exists but is empty.")
    else:
        warnings.append(f"No metrics file found for {project}.")

    confusion_runs: dict[int, pd.DataFrame] = {}
    if CONF_DIR.is_dir():
        for path in sorted(CONF_DIR.glob(f"directed_{project}_results*.csv")):
            match = _CONF_NAME.match(path.name)
            if not match:
                continue
            frame = load_confusion_matrix(path)
            if frame is None:
                warnings.append(f"{path.name} is empty or unreadable.")
                continue
            confusion_runs[int(match.group(2))] = frame

    return ProjectResults(
        project=project,
        graph_type=graph_type,
        metric_runs=metric_runs,
        confusion_runs=confusion_runs,
        source_metrics_file=metrics_path if metrics_path.is_file() else None,
        warnings=warnings,
    )


def load_all_projects() -> dict[str, ProjectResults]:
    return {name: load_project(name) for name in discover_projects()}


def load_reference_results() -> pd.DataFrame:
    if not REFERENCE_PATH.is_file() or REFERENCE_PATH.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(REFERENCE_PATH)
    if frame.empty:
        return frame
    frame.columns = [c.strip().lower() for c in frame.columns]
    return frame.dropna(how="all")


def list_changeset_projects() -> list[str]:
    folder = DATA_ROOT / "0" / "ChangeSets"
    if not folder.is_dir():
        return []
    return sorted(path.stem for path in folder.glob("*.csv"))


def load_raw_changesets(project: str, shuffle: int) -> list[list[str]] | None:
    path = DATA_ROOT / str(shuffle) / "ChangeSets" / f"{project}.csv"
    if not path.is_file():
        return None
    rows: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            cells = [cell.strip() for cell in line.strip().split(",") if cell.strip()]
            if len(cells) >= 2:
                rows.append(cells)
    return rows or None


def filter_changesets_like_authors(changes: list[list[str]], max_commits: int = 1000) -> list[list[str]]:
    """Replicate TGN_model.py filtering without executing the model."""
    if len(changes) < 4:
        return []
    train_changes = changes[: int(len(changes) * 0.5)]
    sizes = [len(row) for row in train_changes[1:]]
    if not sizes:
        return []
    p90 = float(np.percentile(sizes, 90))
    filtered = [row for row in changes if 2 < len(row) < p90]
    return filtered[:max_commits]
