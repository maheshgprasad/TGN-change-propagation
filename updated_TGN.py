"""Directed co-change prediction, optimized for Linux and WSL.

Example:
    python3 "Pasted code.py" --data-root ./ShuffledData --results-root ./Results
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from pathlib import Path

# Set these before importing TensorFlow. They reduce log noise and prevent WSL
# CPU thread oversubscription; override either variable in the shell if needed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
# TensorFlow also uses these pools on CPU builds. Keeping them aligned avoids
# oversubscription under WSL, while still allowing shell-level overrides.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault(
    "TF_NUM_INTEROP_THREADS",
    str(max(1, int(os.environ["OMP_NUM_THREADS"]) // 2)),
)

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM


DEFAULT_PROJECTS = (
    "alamofire", "ant", "cassandra", "cassandrawebsite", "flutter", "gephi",
    "hbase", "lucene", "laravel", "monitorcontrol", "pydriller", "react",
    "rocketmqclients", "spark", "wwwsite",
)

PROJECT_CONFIG = {
    "alamofire": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "ant": {
        "mu": 0.005, "rho": 95, "l1": 16, "l2": 16,
        "epochs": 7, "optimizer": "nadam",
    },
    "cassandra": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "cassandrawebsite": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "flutter": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "gephi": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "hbase": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "laravel": {
        "mu": 0.005, "rho": 95, "l1": 64, "l2": 2,
        "epochs": 14, "optimizer": "rmsprop",
    },
    "lucene": {
        "mu": 0.1, "rho": 95, "l1": 64, "l2": 4,
        "epochs": 11, "optimizer": "nadam",
    },
    "monitorcontrol": {
        "mu": 0.005, "rho": 95, "l1": 4, "l2": 4,
        "epochs": 5, "optimizer": "adam",
    },
    "pydriller": {
        "mu": 0.1, "rho": 75, "l1": 4, "l2": 4,
        "epochs": 6, "optimizer": "nadam",
    },
    "react": {
        "mu": 0.005, "rho": 95, "l1": 8, "l2": 8,
        "epochs": 15, "optimizer": "sgd",
    },
    "rocketmqclients": {
        "mu": 0.2, "rho": 60, "l1": 32, "l2": 32,
        "epochs": 15, "optimizer": "adagrad",
    },
    "spark": {
        "mu": 0.005, "rho": 60, "l1": 8, "l2": 4,
        "epochs": 7, "optimizer": "rmsprop",
    },
    "wwwsite": {
        "mu": 0.3, "rho": 50, "l1": 8, "l2": 32,
        "epochs": 9, "optimizer": "adam",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("ShuffledData"))
    parser.add_argument("--results-root", type=Path, default=Path("Results"))
    parser.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument(
        "--shuffle-indices",
        nargs="+",
        type=int,
        default=None,
        help="Specific shuffle indices and execution order, e.g. 3 1 4 0 2",
    )
    parser.add_argument("--max-commits", type=int, default=1000)
    #parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_tensorflow(threads: int, seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    # This workload creates many short-lived models. XLA compilation overhead
    # and its cached executables can otherwise consume substantial host memory.
    tf.config.optimizer.set_jit(False)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(max(1, threads // 2))
    except RuntimeError:
        pass  # TensorFlow was already initialized.

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(
        f"TensorFlow device: {'GPU' if gpus else 'CPU'}; "
        f"threads: {threads}; XLA: disabled"
    )


def load_changes(path: Path, max_commits: int) -> list[list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[cell.strip() for cell in row if cell.strip()] for row in csv.reader(handle)]
    rows = [row for row in rows if len(row) >= 2]
    if len(rows) < 4:
        raise ValueError(f"Not enough valid commits in {path}")

    initial_train = rows[: len(rows) // 2]
    commit_sizes = [len(row) for row in initial_train[1:]]
    p90 = np.percentile(commit_sizes, 90) if commit_sizes else float("inf")
    filtered = [row for row in rows if 2 < len(row) < p90]
    return filtered[:max_commits]


def count_in_range(values: list[int], start: int, end: int) -> int:
    return bisect_right(values, end) - bisect_left(values, start)


def cochangeability(
    file_a: str,
    file_b: str,
    end_time: int,
    changes_of_file: dict[str, list[int]],
    changes_set: dict[str, set[int]],
) -> float:
    """Preserve the original directed score: common(A,B) / changes(A)."""
    a_values = changes_of_file[file_a]
    b_values = changes_of_file[file_b]
    start = max(a_values[0], b_values[0])
    a_left = bisect_left(a_values, start)
    a_right = bisect_right(a_values, end_time)
    denominator = a_right - a_left
    if denominator == 0:
        return 0.0

    b_left = bisect_left(b_values, start)
    b_right = bisect_right(b_values, end_time)

    # Count the intersection by scanning whichever in-range history is shorter.
    # This produces the same directed score but substantially reduces work for
    # files with very uneven change frequencies.
    if denominator <= b_right - b_left:
        common = sum(
            index in changes_set[file_b]
            for index in a_values[a_left:a_right]
        )
    else:
        common = sum(
            index in changes_set[file_a]
            for index in b_values[b_left:b_right]
        )
    return common / denominator


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def create_model(layer_1: int, layer_2: int, optimizer: str) -> Sequential:
    model = Sequential([
        Input(shape=(1, 1)),
        LSTM(layer_1, activation="tanh", return_sequences=True),
        LSTM(layer_2, activation="tanh"),
        Dense(1, activation="sigmoid"),
    ])
    optimizer_instance = tf.keras.optimizers.get(optimizer)
    # TensorFlow 2.14 optimizers may enable XLA for update steps independently
    # of Model.compile(), so disable it on both the optimizer and model.
    if hasattr(optimizer_instance, "jit_compile"):
        optimizer_instance.jit_compile = False
    model.compile(
        optimizer=optimizer_instance,
        loss="binary_crossentropy",
        jit_compile=False,
    )
    return model


def append_csv(path: Path, row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(row)


def run_experiment(
    changes: list[list[str]],
    project: str,
    shuffle: int,
    results_root: Path,
    verbose: bool,
) -> None:
    if len(changes) < 4:
        print(f"Skipping {project}/{shuffle}: fewer than four filtered commits")
        return

    split = len(changes) // 2
    train_changes = changes[:split]
 #   layer_1 = layer_2 = 4
 #   cochange_cutoff = 0.005
 #   predicted_size_percentile = 95
 #   optimizer = "adam"
    config = PROJECT_CONFIG[project]
    
    layer_1 = config["l1"]
    layer_2 = config["l2"]
    cochange_cutoff = config["mu"]
    predicted_size_percentile = config["rho"]
    optimizer = config["optimizer"]
    epochs = config["epochs"]

    changes_of_file: dict[str, list[int]] = defaultdict(list)
    changes_set: dict[str, set[int]] = defaultdict(set)
    for commit_index, commit in enumerate(train_changes):
        for filename in commit[1:]:
            changes_of_file[filename].append(commit_index)
            changes_set[filename].add(commit_index)

    # Keep only the current graph plus compact change-history snapshots. The
    # original retained a full graph copy for every commit, consuming much more RAM.
    node_history: list[set[str]] = [set(commit[1:]) for commit in train_changes]
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    first_files = train_changes[0][1:]
    for source in first_files:
        for target in first_files:
            if source != target:
                graph[source][target] = 1.0

    known_files: set[str] = set(first_files)
    for commit_index, commit in enumerate(train_changes[1:], start=1):
        current_files = commit[1:]
        known_files.update(current_files)
        for source in current_files:
            # Preserve the source node even when none of its outgoing edges
            # currently meet the cutoff. This keeps seed selection unchanged.
            source_edges = graph[source]
            for target in known_files:
                if source != target:
                    score = cochangeability(
                        source, target, commit_index, changes_of_file, changes_set
                    )
                    # Prediction already ignores scores below this cutoff. Not
                    # storing them prevents the graph from approaching O(F^2)
                    # Python dictionary entries as unique file count grows.
                    if score >= cochange_cutoff:
                        source_edges[target] = score
                    else:
                        source_edges.pop(target, None)

    train_sizes = [len(commit) for commit in train_changes[1:]]
    predicted_size = int(np.percentile(train_sizes, predicted_size_percentile))
    metric_names = ("sensitivity", "specificity", "ppv", "gmean", "fmeasure", "accuracy", "mcc", "auc")
    metrics: dict[str, list[float]] = {name: [] for name in metric_names}
    confusion_rows: list[list[int]] = []
    models_created = 0

    for test_commit_index in range(split, len(changes)):
        actual = changes[test_commit_index][1:]
        seed = next((name for name in actual if name in graph), None)
        if seed is None:
            continue

        queue = deque([seed])
        predicted = [seed]
        predicted_lookup = {seed}

        while queue and len(predicted) < predicted_size:
            current = queue.pop()
            neighbors = graph.get(current, {})
            for neighbor in sorted(neighbors):
                if (neighbors[neighbor] < cochange_cutoff or neighbor in predicted_lookup
                        or len(predicted) >= predicted_size):
                    continue

                x_values: list[float] = []
                y_values: list[float] = []
                # Reconstruct historical co-change scores only for candidate pairs,
                # instead of retaining every full graph snapshot.
                for history_index in range(1, len(node_history) - 1):
                    if current in node_history[history_index]:
                        if (changes_of_file[current][0] > history_index - 1
                                or changes_of_file[neighbor][0] > history_index - 1):
                            continue
                        score = cochangeability(
                            current, neighbor, history_index - 1, changes_of_file, changes_set
                        )
                        x_values.append(score)
                        y_values.append(float(neighbor in node_history[history_index]))
                if not x_values:
                    continue

                x_train = np.asarray(x_values, dtype=np.float32).reshape(-1, 1, 1)
                y_train = np.asarray(y_values, dtype=np.float32)
                x_test = np.array([[[neighbors[neighbor]]]], dtype=np.float32)
                model = create_model(layer_1, layer_2, optimizer)
                model.fit(x_train, y_train, batch_size=len(x_train), epochs=epochs, verbose=0)
                probability = float(model(x_test, training=False).numpy()[0, 0])
                del model
                tf.keras.backend.clear_session()
                models_created += 1
                # Force collection periodically rather than after every model,
                # avoiding both unbounded Python garbage and excessive GC cost.
                if models_created % 100 == 0:
                    gc.collect()

                if probability > 0.5:
                    queue.append(neighbor)
                    predicted.append(neighbor)
                    predicted_lookup.add(neighbor)

        actual_lookup = set(actual)
        tp = len(actual_lookup & predicted_lookup)
        fn = len(actual_lookup - predicted_lookup)
        fp = len(predicted_lookup - actual_lookup)
        tn = max(0, len(known_files) - tp - fp - fn)
        confusion_rows.append([tp, tn, fp, fn])

        sensitivity = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        values = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "gmean": (sensitivity * specificity) ** 0.5,
            "fmeasure": safe_div(2 * ppv * sensitivity, ppv + sensitivity),
            "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
            "mcc": safe_div(tp * tn - fp * fn, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5),
            "auc": 0.5 * (sensitivity + specificity),
        }
        for name, value in values.items():
            metrics[name].append(value)

        # Incrementally update history and graph with this test commit.
        for filename in actual:
            changes_of_file[filename].append(test_commit_index)
            changes_set[filename].add(test_commit_index)
        node_history.append(set(actual))
        known_files.update(actual)
        for source in actual:
            source_edges = graph[source]
            for target in known_files:
                if source != target:
                    score = cochangeability(
                        source, target, test_commit_index, changes_of_file, changes_set
                    )
                    if score >= cochange_cutoff:
                        source_edges[target] = score
                    else:
                        source_edges.pop(target, None)
        if verbose:
            print(f"  commit {test_commit_index + 1}/{len(changes)}: predicted={len(predicted)}, actual={len(actual)}")

    confusion_path = results_root / "ConfMatrix" / f"directed_{project}_results_{shuffle}.csv"
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    with confusion_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(confusion_rows)

    means = [float(np.mean(metrics[name])) if metrics[name] else 0.0 for name in metric_names]
    append_csv(
        results_root / "Metrics" / f"directed_{project}_results.csv",
        ["directed", project, *means, layer_1, layer_2, cochange_cutoff,
         predicted_size_percentile, optimizer, epochs],
    )


def main() -> int:
    args = parse_args()
    configure_tensorflow(max(1, args.threads), args.seed)
    shuffle_indices = (
        args.shuffle_indices
        if args.shuffle_indices is not None
        else range(args.shuffles)
    )

    for project in args.projects:
        print(f"Project: {project}")
        for shuffle in shuffle_indices:
            input_path = args.data_root / str(shuffle) / "ChangeSets" / f"{project}.csv"
            try:
                changes = load_changes(input_path, args.max_commits)
                print(f" Shuffle {shuffle}: {len(changes)} filtered commits")
                run_experiment(changes, project, shuffle, args.results_root, args.verbose)
            except (FileNotFoundError, ValueError) as error:
                print(f" Skipping shuffle {shuffle}: {error}")
            finally:
                # Release per-shuffle data before the next dataset is loaded.
                if "changes" in locals():
                    del changes
                tf.keras.backend.clear_session()
                gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


