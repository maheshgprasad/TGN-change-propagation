from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from core import (
    PROJECT_CONFIG,
    build_samples,
    evaluate_dfs,
    load_changes,
    select_threshold,
    train_model,
    write_csv,
    write_json,
)
from model import ModelConfig


def args_parser():
    p = argparse.ArgumentParser(
        description="Clean Phase-1 shared temporal-attention experiment"
    )
    p.add_argument(
        "--project",
        choices=sorted(PROJECT_CONFIG),
        required=True,
    )
    p.add_argument("--shuffle", type=int, default=0)
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("ShuffledData"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("Phase1CleanResults"),
    )
    p.add_argument(
        "--max-commits",
        type=int,
        default=1000,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--history-k",
        type=int,
        default=20,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=20,
    )
    p.add_argument(
        "--threshold-start",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--threshold-stop",
        type=float,
        default=0.95,
    )
    p.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Skip calibration and use a fixed threshold",
    )
    return p.parse_args()


def main():
    args = args_parser()
    cfg = PROJECT_CONFIG[args.project]

    model_cfg = ModelConfig(
        history_k=args.history_k,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )

    input_path = (
        args.data_root
        / str(args.shuffle)
        / "ChangeSets"
        / f"{args.project}.csv"
    )

    out = (
        args.output_root
        / f"{args.project}_shuffle_{args.shuffle}"
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    changes = load_changes(
        input_path,
        args.max_commits,
    )

    split = len(changes) // 2
    train_end = int(split * 0.8)
    val_start = train_end
    val_end = split

    train_sizes = [
        len(c)
        for c in changes[:split][1:]
    ]

    predicted_size = int(
        np.percentile(
            train_sizes,
            cfg["rho"],
        )
    )

    print(
        f"project={args.project} "
        f"filtered_commits={len(changes)} "
        f"baseline_split={split}"
    )
    print(
        f"model_train=0..{train_end - 1} "
        f"validation={val_start}..{val_end - 1} "
        f"test={split}..{len(changes) - 1}"
    )
    print(
        f"mu={cfg['mu']} "
        f"rho={cfg['rho']} "
        f"cap={predicted_size}"
    )

    t0 = time.perf_counter()

    samples = build_samples(
        changes,
        train_end,
        val_end,
        cfg["mu"],
        model_cfg.history_k,
    )

    t1 = time.perf_counter()

    model, transform, train_info = train_model(
        samples,
        model_cfg,
        args.seed,
        args.device,
    )

    t2 = time.perf_counter()

    if args.threshold is None:
        thresholds = np.arange(
            args.threshold_start,
            args.threshold_stop + 1e-9,
            args.threshold_step,
        )

        selected, grid = select_threshold(
            changes,
            val_start,
            val_end,
            cfg["mu"],
            predicted_size,
            model,
            transform,
            model_cfg.history_k,
            args.device,
            thresholds,
        )

        write_csv(
            out / "threshold_grid.csv",
            grid,
        )
    else:
        selected = float(args.threshold)

    t3 = time.perf_counter()

    final = evaluate_dfs(
        changes,
        split,
        len(changes),
        cfg["mu"],
        predicted_size,
        selected,
        model,
        transform,
        model_cfg.history_k,
        args.device,
        trace=True,
    )

    t4 = time.perf_counter()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "feature_transform": transform.to_dict(),
            "threshold": selected,
            "project": args.project,
            "shuffle": args.shuffle,
            "mu": cfg["mu"],
            "rho": cfg["rho"],
            "predicted_size": predicted_size,
        },
        out / "model.pt",
    )

    write_csv(
        out / "test_commit_predictions.csv",
        final["commit_rows"],
    )

    write_csv(
        out / "candidate_trace.csv",
        final["candidate_rows"],
    )

    write_json(
        out / "summary.json",
        {
            "project": args.project,
            "shuffle": args.shuffle,
            "filtered_commits": len(changes),
            "baseline_split": split,
            "train_end": train_end,
            "validation_start": val_start,
            "validation_end": val_end,
            "mu": cfg["mu"],
            "rho": cfg["rho"],
            "predicted_size_cap": predicted_size,
            "selected_threshold": selected,
            "training": train_info,
            "test_means": final["means"],
            "behavior": final["behavior"],
            "timing_seconds": {
                "sample_build": t1 - t0,
                "training": t2 - t1,
                "calibration": t3 - t2,
                "test": t4 - t3,
                "total": t4 - t0,
            },
        },
    )

    print(
        f"selected_threshold={selected:.2f}"
    )
    print(
        "test_means=",
        final["means"],
    )
    print(
        "behavior=",
        final["behavior"],
    )
    print(
        f"total_seconds={t4 - t0:.2f}"
    )


if __name__ == "__main__":
    main()
