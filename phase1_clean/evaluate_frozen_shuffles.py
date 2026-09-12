from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core import (
    FeatureTransform,
    PROJECT_CONFIG,
    evaluate_dfs,
    load_changes,
    write_csv,
    write_json,
)
from model import ModelConfig, TemporalAttentionScorer


PROJECTS = tuple(PROJECT_CONFIG.keys())


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate shuffle-0 trained clean-attention models on shuffles 1-4 "
            "without retraining or recalibration."
        )
    )
    p.add_argument(
        "--projects",
        nargs="+",
        default=list(PROJECTS),
        choices=sorted(PROJECTS),
    )
    p.add_argument(
        "--shuffles",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("ShuffledData"),
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("Phase1CleanResults"),
    )
    p.add_argument(
        "--baseline-conf-root",
        type=Path,
        default=Path("Results/ConfMatrix"),
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-commits", type=int, default=1000)
    p.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    return p.parse_args()


def load_frozen_model(project: str, results_root: Path, device: str):
    artifact_path = results_root / f"{project}_shuffle_0" / "model.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing shuffle-0 frozen model: {artifact_path}")

    payload = torch.load(artifact_path, map_location=device)

    cfg = ModelConfig(**payload["model_config"])
    transform_data = payload["feature_transform"]

    transform = FeatureTransform()
    transform.median = np.asarray(transform_data["median"], dtype=np.float64)
    transform.mean = np.asarray(transform_data["mean"], dtype=np.float64)
    transform.std = np.asarray(transform_data["std"], dtype=np.float64)

    feature_dim = len(transform_data["mean"]) + len(transform_data["recency_indexes"])

    model = TemporalAttentionScorer(
        feature_dim=feature_dim,
        model_dim=cfg.model_dim,
        heads=cfg.heads,
        time_dim=cfg.time_dim,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return {
        "model": model,
        "transform": transform,
        "config": cfg,
        "threshold": float(payload["threshold"]),
        "mu": float(payload["mu"]),
        "rho": float(payload["rho"]),
        "predicted_size": int(payload["predicted_size"]),
        "artifact": artifact_path,
    }


def baseline_metrics_from_rows(frame: pd.DataFrame) -> dict[str, float]:
    rows = []
    for row in frame.itertuples(index=False):
        tp = float(row.tp)
        tn = float(row.tn)
        fp = float(row.fp)
        fn = float(row.fn)

        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        ppv = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2 * ppv * sensitivity / (ppv + sensitivity)
            if ppv + sensitivity
            else 0.0
        )
        denom = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = (tp * tn - fp * fn) / denom if denom else 0.0
        rows.append(
            {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "f1": f1,
                "mcc": mcc,
                "legacy_auc": 0.5 * (sensitivity + specificity),
            }
        )

    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def load_baseline_confusion(
    path: Path,
    expected_rows: int,
) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return None, "missing"

    frame = pd.read_csv(
        path,
        header=None,
        names=["tp", "tn", "fp", "fn"],
        usecols=[0, 1, 2, 3],
    )
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all").fillna(0)

    if len(frame) == expected_rows:
        return frame, None

    if expected_rows > 0 and len(frame) > expected_rows and len(frame) % expected_rows == 0:
        # Some baseline CSVs were append-written more than once. Keep the first
        # logical run only when the file is an exact repetition multiple.
        return frame.iloc[:expected_rows].copy(), f"trimmed_from_{len(frame)}"

    return None, f"row_mismatch_{len(frame)}_vs_{expected_rows}"


def comparison_row(
    project: str,
    shuffle: int,
    clean_summary: dict,
    baseline_conf_root: Path,
) -> dict:
    baseline_path = (
        baseline_conf_root / f"directed_{project}_results_{shuffle}.csv"
    )
    expected = int(clean_summary["behavior"]["evaluated_commits"])
    baseline_frame, baseline_note = load_baseline_confusion(
        baseline_path,
        expected,
    )

    row = {
        "project": project,
        "shuffle": shuffle,
        "threshold": clean_summary["selected_threshold"],
        "prediction_cap": clean_summary["predicted_size_cap"],
        "evaluated_commits": expected,
        "clean_f1": clean_summary["test_means"]["f1"],
        "clean_mcc": clean_summary["test_means"]["mcc"],
        "clean_recall": clean_summary["test_means"]["sensitivity"],
        "clean_ppv": clean_summary["test_means"]["ppv"],
        "clean_mean_fp": clean_summary["behavior"]["mean_fp"],
        "clean_mean_predicted_size": clean_summary["behavior"]["mean_predicted_size"],
        "clean_cap_hit_rate": clean_summary["behavior"]["cap_hit_rate"],
        "clean_test_seconds": clean_summary["timing_seconds"]["test"],
        "baseline_note": baseline_note or "",
        "comparable": baseline_frame is not None,
    }

    if baseline_frame is None:
        for key in (
            "baseline_f1",
            "baseline_mcc",
            "baseline_recall",
            "baseline_ppv",
            "delta_f1",
            "delta_mcc",
            "delta_recall",
            "delta_ppv",
        ):
            row[key] = np.nan
        return row

    baseline = baseline_metrics_from_rows(baseline_frame)
    row.update(
        {
            "baseline_f1": baseline["f1"],
            "baseline_mcc": baseline["mcc"],
            "baseline_recall": baseline["sensitivity"],
            "baseline_ppv": baseline["ppv"],
            "delta_f1": row["clean_f1"] - baseline["f1"],
            "delta_mcc": row["clean_mcc"] - baseline["mcc"],
            "delta_recall": row["clean_recall"] - baseline["sensitivity"],
            "delta_ppv": row["clean_ppv"] - baseline["ppv"],
        }
    )
    return row


def evaluate_one(
    project: str,
    shuffle: int,
    frozen: dict,
    args,
) -> tuple[dict, dict]:
    input_path = (
        args.data_root
        / str(shuffle)
        / "ChangeSets"
        / f"{project}.csv"
    )
    changes = load_changes(input_path, args.max_commits)
    split = len(changes) // 2

    t0 = time.perf_counter()
    result = evaluate_dfs(
        changes=changes,
        start=split,
        end=len(changes),
        mu=frozen["mu"],
        predicted_size=frozen["predicted_size"],
        threshold=frozen["threshold"],
        model=frozen["model"],
        transform=frozen["transform"],
        history_k=frozen["config"].history_k,
        device=args.device,
        trace=False,
    )
    elapsed = time.perf_counter() - t0

    out_dir = (
        args.results_root
        / "frozen_shuffles"
        / f"{project}_shuffle_{shuffle}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "test_commit_predictions.csv",
        result["commit_rows"],
    )

    summary = {
        "project": project,
        "shuffle": shuffle,
        "evaluation_mode": "frozen_shuffle0_model",
        "source_model": str(frozen["artifact"]),
        "filtered_commits": len(changes),
        "baseline_split": split,
        "mu": frozen["mu"],
        "rho": frozen["rho"],
        "predicted_size_cap": frozen["predicted_size"],
        "selected_threshold": frozen["threshold"],
        "test_means": result["means"],
        "behavior": result["behavior"],
        "timing_seconds": {
            "test": elapsed,
        },
    }
    write_json(out_dir / "summary.json", summary)

    return summary, comparison_row(
        project,
        shuffle,
        summary,
        args.baseline_conf_root,
    )


def load_shuffle0_summary(project: str, results_root: Path) -> dict:
    path = results_root / f"{project}_shuffle_0" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shuffle-0 summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def project_aggregate(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    output = []

    for project, group in frame.groupby("project", sort=True):
        comparable = group[group["comparable"] == True].copy()
        row = {
            "project": project,
            "evaluations": int(len(group)),
            "comparable_evaluations": int(len(comparable)),
            "f1_wins": int((comparable["delta_f1"] > 0).sum()),
            "mcc_wins": int((comparable["delta_mcc"] > 0).sum()),
            "mean_clean_f1": float(group["clean_f1"].mean()),
            "mean_clean_mcc": float(group["clean_mcc"].mean()),
            "mean_clean_recall": float(group["clean_recall"].mean()),
            "mean_clean_ppv": float(group["clean_ppv"].mean()),
        }

        if not comparable.empty:
            row.update(
                {
                    "mean_baseline_f1": float(comparable["baseline_f1"].mean()),
                    "mean_baseline_mcc": float(comparable["baseline_mcc"].mean()),
                    "mean_delta_f1": float(comparable["delta_f1"].mean()),
                    "mean_delta_mcc": float(comparable["delta_mcc"].mean()),
                    "mean_delta_recall": float(comparable["delta_recall"].mean()),
                    "mean_delta_ppv": float(comparable["delta_ppv"].mean()),
                }
            )
        else:
            row.update(
                {
                    "mean_baseline_f1": np.nan,
                    "mean_baseline_mcc": np.nan,
                    "mean_delta_f1": np.nan,
                    "mean_delta_mcc": np.nan,
                    "mean_delta_recall": np.nan,
                    "mean_delta_ppv": np.nan,
                }
            )

        output.append(row)

    return output


def main():
    args = parse_args()
    all_rows: list[dict] = []
    failures: list[str] = []

    for project in args.projects:
        print("\n" + "=" * 80)
        print(f"{project}: loading frozen shuffle-0 model")
        print("=" * 80)

        try:
            frozen = load_frozen_model(
                project,
                args.results_root,
                args.device,
            )

            shuffle0 = load_shuffle0_summary(
                project,
                args.results_root,
            )
            all_rows.append(
                comparison_row(
                    project,
                    0,
                    shuffle0,
                    args.baseline_conf_root,
                )
            )

            for shuffle in args.shuffles:
                print(
                    f"Evaluating {project} shuffle {shuffle} "
                    f"with frozen threshold={frozen['threshold']:.2f}"
                )
                summary, row = evaluate_one(
                    project,
                    shuffle,
                    frozen,
                    args,
                )
                all_rows.append(row)
                print(
                    f"  F1={summary['test_means']['f1']:.3f} "
                    f"MCC={summary['test_means']['mcc']:.3f} "
                    f"test={summary['timing_seconds']['test']:.2f}s"
                )

        except Exception as exc:
            failures.append(f"{project}: {exc}")
            print(f"FAILED: {project}: {exc}")
            if not args.continue_on_error:
                raise

    aggregate_dir = args.results_root / "frozen_shuffles"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        aggregate_dir / "all_shuffles_comparison.csv",
        all_rows,
    )
    write_csv(
        aggregate_dir / "project_summary.csv",
        project_aggregate(all_rows),
    )
    write_json(
        aggregate_dir / "run_manifest.json",
        {
            "evaluation_mode": "frozen_shuffle0_model",
            "evaluated_shuffles": [0] + list(args.shuffles),
            "projects": list(args.projects),
            "failures": failures,
            "comparison_rows": len(all_rows),
        },
    )

    print("\nWrote:")
    print(aggregate_dir / "all_shuffles_comparison.csv")
    print(aggregate_dir / "project_summary.csv")
    print(aggregate_dir / "run_manifest.json")

    if failures:
        print("\nFailures:")
        for item in failures:
            print(" -", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
