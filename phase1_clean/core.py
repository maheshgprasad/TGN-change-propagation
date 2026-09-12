from __future__ import annotations

import csv
import json
from bisect import bisect_left
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, TemporalAttentionScorer


PROJECT_CONFIG = {
    "alamofire": {"mu": 0.005, "rho": 95},
    "ant": {"mu": 0.005, "rho": 95},
    "cassandra": {"mu": 0.005, "rho": 95},
    "laravel": {"mu": 0.005, "rho": 95},
    "lucene": {"mu": 0.1, "rho": 95},
    "monitorcontrol": {"mu": 0.005, "rho": 95},
    "pydriller": {"mu": 0.1, "rho": 75},
    "react": {"mu": 0.005, "rho": 95},
    "rocketmqclients": {"mu": 0.2, "rho": 60},
    "spark": {"mu": 0.005, "rho": 60},
}

BASE_FEATURES = (
    "cochange_prior",
    "pair_count_prior",
    "source_freq_prior",
    "target_freq_prior",
    "source_recency_commits",
    "target_recency_commits",
    "pair_recency_commits",
    "source_recency_seconds",
    "target_recency_seconds",
    "pair_recency_seconds",
    "graph_edge_exists_prior",
    "known_files_prior",
)
RECENCY_INDEXES = (4, 5, 6, 7, 8, 9)
COUNTLIKE_INDEXES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_timestamp(text: str) -> float:
    return datetime.fromisoformat(text.strip()).timestamp()


def load_changes(path: Path, max_commits: int = 1000) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[c.strip() for c in row if c.strip()] for row in csv.reader(handle)]
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < 4:
        raise ValueError("Not enough commits")

    raw_first_half = rows[: len(rows) // 2]
    sizes = [len(r) for r in raw_first_half[1:]]
    p90 = np.percentile(sizes, 90) if sizes else float("inf")
    filtered = [r for r in rows if 2 < len(r) < p90]
    return filtered[:max_commits]


class TemporalState:
    """Historical state strictly before the current commit."""

    def __init__(self, mu: float):
        self.mu = mu
        self.known: set[str] = set()
        self.file_count: dict[str, int] = defaultdict(int)
        self.file_commits: dict[str, list[int]] = defaultdict(list)
        self.pair_count: dict[tuple[str, str], int] = defaultdict(int)
        self.last_file_commit: dict[str, int] = {}
        self.last_pair_commit: dict[tuple[str, str], int] = {}
        self.last_file_time: dict[str, float] = {}
        self.last_pair_time: dict[tuple[str, str], float] = {}

    def cochange(self, source: str, target: str) -> float:
        if source not in self.file_commits or target not in self.file_commits:
            return 0.0
        start = max(self.file_commits[source][0], self.file_commits[target][0])
        src = self.file_commits[source]
        denom = len(src) - bisect_left(src, start)
        if denom <= 0:
            return 0.0
        return self.pair_count.get((source, target), 0) / denom

    def features(self, source: str, target: str, commit_idx: int, timestamp: float) -> np.ndarray:
        def delta_commit(last: int | None) -> float:
            return np.nan if last is None else float(commit_idx - last)

        def delta_time(last: float | None) -> float:
            return np.nan if last is None else max(0.0, float(timestamp - last))

        edge_w = self.cochange(source, target)
        return np.asarray(
            [
                edge_w,
                float(self.pair_count.get((source, target), 0)),
                float(self.file_count.get(source, 0)),
                float(self.file_count.get(target, 0)),
                delta_commit(self.last_file_commit.get(source)),
                delta_commit(self.last_file_commit.get(target)),
                delta_commit(self.last_pair_commit.get((source, target))),
                delta_time(self.last_file_time.get(source)),
                delta_time(self.last_file_time.get(target)),
                delta_time(self.last_pair_time.get((source, target))),
                1.0 if edge_w >= self.mu else 0.0,
                float(len(self.known)),
            ],
            dtype=np.float32,
        )

    def neighbors(self, source: str) -> list[str]:
        if source not in self.known:
            return []
        out = []
        for target in self.known:
            if target != source and self.cochange(source, target) >= self.mu:
                out.append(target)
        return sorted(out)

    def update(self, files: list[str], commit_idx: int, timestamp: float) -> None:
        uniq = list(dict.fromkeys(files))
        for source in uniq:
            self.file_count[source] += 1
            self.file_commits[source].append(commit_idx)
            self.last_file_commit[source] = commit_idx
            self.last_file_time[source] = timestamp

        for source in uniq:
            for target in uniq:
                if source == target:
                    continue
                key = (source, target)
                self.pair_count[key] += 1
                self.last_pair_commit[key] = commit_idx
                self.last_pair_time[key] = timestamp

        self.known.update(uniq)


@dataclass
class SampleStore:
    x_raw: np.ndarray
    y: np.ndarray
    hist_idx: np.ndarray
    hist_age: np.ndarray
    same_target: np.ndarray
    pos_raw: np.ndarray
    split: np.ndarray


def build_samples(
    changes: list[list[str]],
    train_end: int,
    val_end: int,
    mu: float,
    history_k: int,
) -> SampleStore:
    state = TemporalState(mu)
    positive_by_source: dict[str, list[int]] = defaultdict(list)
    positive_features: list[np.ndarray] = []
    positive_times: list[float] = []
    positive_targets: list[str] = []

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    hist_rows: list[np.ndarray] = []
    age_rows: list[np.ndarray] = []
    same_rows: list[np.ndarray] = []
    split_rows: list[int] = []

    for commit_idx, commit in enumerate(changes[:val_end]):
        ts = parse_timestamp(commit[0])
        files = list(dict.fromkeys(commit[1:]))
        actual = set(files)
        split_flag = 0 if commit_idx < train_end else 1

        pending_positive_events: list[tuple[str, str, np.ndarray, float]] = []

        for source in files:
            if source not in state.known:
                continue

            candidates = sorted(state.known - {source})
            hist_source = positive_by_source.get(source, [])[-history_k:]

            for target in candidates:
                feat = state.features(source, target, commit_idx, ts)
                label = 1 if target in actual else 0

                x_rows.append(feat)
                y_rows.append(label)
                split_rows.append(split_flag)

                idxs = np.full(history_k, -1, dtype=np.int32)
                ages = np.zeros(history_k, dtype=np.float32)
                same = np.zeros(history_k, dtype=np.float32)

                tail = hist_source[-history_k:]
                start = history_k - len(tail)
                for j, pos_idx in enumerate(tail, start=start):
                    idxs[j] = pos_idx
                    ages[j] = max(0.0, ts - positive_times[pos_idx])
                    same[j] = 1.0 if positive_targets[pos_idx] == target else 0.0

                hist_rows.append(idxs)
                age_rows.append(ages)
                same_rows.append(same)

                if label == 1:
                    pending_positive_events.append((source, target, feat, ts))

        for source, target, feat, event_ts in pending_positive_events:
            pos_idx = len(positive_features)
            positive_features.append(feat)
            positive_times.append(event_ts)
            positive_targets.append(target)
            positive_by_source[source].append(pos_idx)

        state.update(files, commit_idx, ts)

    if not x_rows:
        raise ValueError("No eligible pair-event samples were produced")

    return SampleStore(
        x_raw=np.stack(x_rows),
        y=np.asarray(y_rows, dtype=np.float32),
        hist_idx=np.stack(hist_rows),
        hist_age=np.stack(age_rows),
        same_target=np.stack(same_rows),
        pos_raw=np.stack(positive_features)
        if positive_features
        else np.zeros((0, len(BASE_FEATURES)), np.float32),
        split=np.asarray(split_rows, dtype=np.int8),
    )


class FeatureTransform:
    def __init__(self):
        self.median = None
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> "FeatureTransform":
        x = x.astype(np.float64, copy=True)
        med = np.zeros(x.shape[1], dtype=np.float64)

        for j in range(x.shape[1]):
            finite = x[:, j][np.isfinite(x[:, j])]
            med[j] = float(np.median(finite)) if finite.size else 0.0

        filled = np.where(np.isfinite(x), x, med)

        for j in COUNTLIKE_INDEXES:
            filled[:, j] = np.log1p(np.maximum(filled[:, j], 0.0))

        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std[std < 1e-8] = 1.0

        self.median = med
        self.mean = mean
        self.std = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None:
            raise RuntimeError("FeatureTransform must be fit first")

        x = x.astype(np.float64, copy=True)
        missing = ~np.isfinite(x[:, RECENCY_INDEXES])
        filled = np.where(np.isfinite(x), x, self.median)

        for j in COUNTLIKE_INDEXES:
            filled[:, j] = np.log1p(np.maximum(filled[:, j], 0.0))

        z = (filled - self.mean) / self.std
        return np.concatenate(
            [z, missing.astype(np.float64)], axis=1
        ).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "base_features": list(BASE_FEATURES),
            "recency_indexes": list(RECENCY_INDEXES),
        }


class PairDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        hist_idx: np.ndarray,
        hist_age: np.ndarray,
        same_target: np.ndarray,
        pos_x: np.ndarray,
    ):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.hist_idx = torch.from_numpy(hist_idx.astype(np.int64))
        self.hist_age = torch.from_numpy(hist_age)
        self.same_target = torch.from_numpy(same_target)
        self.pos_x = torch.from_numpy(pos_x)
        self.feature_dim = x.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        refs = self.hist_idx[idx]
        mask = refs >= 0
        safe = refs.clamp(min=0)

        if len(self.pos_x):
            hist = self.pos_x[safe].clone()
            hist[~mask] = 0.0
        else:
            hist = torch.zeros(
                (len(refs), self.feature_dim), dtype=torch.float32
            )

        return (
            self.x[idx],
            hist,
            self.hist_age[idx],
            self.same_target[idx],
            mask,
            self.y[idx],
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    samples: SampleStore,
    cfg: ModelConfig,
    seed: int,
    device: str = "cpu",
) -> tuple[TemporalAttentionScorer, FeatureTransform, dict]:
    seed_everything(seed)

    train_mask = samples.split == 0
    val_mask = samples.split == 1

    transform = FeatureTransform().fit(samples.x_raw[train_mask])
    x_all = transform.transform(samples.x_raw)
    pos_x = (
        transform.transform(samples.pos_raw)
        if len(samples.pos_raw)
        else np.zeros((0, x_all.shape[1]), np.float32)
    )

    def make_ds(mask):
        return PairDataset(
            x_all[mask],
            samples.y[mask],
            samples.hist_idx[mask],
            samples.hist_age[mask],
            samples.same_target[mask],
            pos_x,
        )

    train_ds = make_ds(train_mask)
    val_ds = make_ds(val_mask)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = TemporalAttentionScorer(
        feature_dim=x_all.shape[1],
        model_dim=cfg.model_dim,
        heads=cfg.heads,
        time_dim=cfg.time_dim,
        dropout=cfg.dropout,
    ).to(device)

    pos = float(samples.y[train_mask].sum())
    neg = float(train_mask.sum() - pos)
    pos_weight = torch.tensor(
        [neg / max(pos, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    bad = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            cur, hist, age, same, mask, y = [b.to(device) for b in batch]
            optim.zero_grad(set_to_none=True)
            logits = model(cur, hist, age, same, mask)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            train_loss += float(loss.item()) * len(y)
            n_train += len(y)

        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                cur, hist, age, same, mask, y = [b.to(device) for b in batch]
                logits = model(cur, hist, age, same, mask)
                loss = criterion(logits, y)
                val_loss += float(loss.item()) * len(y)
                n_val += len(y)

        train_mean = train_loss / max(n_train, 1)
        val_mean = val_loss / max(n_val, 1)

        history.append(
            {
                "epoch": epoch,
                "train_bce": train_mean,
                "val_bce": val_mean,
            }
        )

        if val_mean < best_val - 1e-6:
            best_val = val_mean
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    return model, transform, {
        "history": history,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "train_positive": int(samples.y[train_mask].sum()),
        "val_positive": int(samples.y[val_mask].sum()),
        "pos_weight": float(pos_weight.item()),
        "best_val_bce": best_val,
    }


class OnlineHistory:
    """Compact positive interaction history used by the attention model."""

    def __init__(self, history_k: int):
        self.k = history_k
        self.by_source: dict[
            str,
            deque[tuple[str, np.ndarray, float]],
        ] = defaultdict(lambda: deque(maxlen=self.k))

    def add_commit(
        self,
        state_before: TemporalState,
        files: list[str],
        commit_idx: int,
        ts: float,
    ) -> None:
        uniq = list(dict.fromkeys(files))
        actual = set(uniq)
        pending = []

        for source in uniq:
            if source not in state_before.known:
                continue
            for target in sorted(state_before.known - {source}):
                if target in actual:
                    pending.append(
                        (
                            source,
                            target,
                            state_before.features(
                                source,
                                target,
                                commit_idx,
                                ts,
                            ),
                            ts,
                        )
                    )

        for source, target, feat, event_ts in pending:
            self.by_source[source].append(
                (target, feat, event_ts)
            )

    def tensor_inputs(
        self,
        source: str,
        target: str,
        current_ts: float,
        transform: FeatureTransform,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        items = list(self.by_source.get(source, []))[-self.k:]

        raw = np.full(
            (self.k, len(BASE_FEATURES)),
            np.nan,
            dtype=np.float32,
        )
        age = np.zeros(self.k, dtype=np.float32)
        same = np.zeros(self.k, dtype=np.float32)
        mask = np.zeros(self.k, dtype=bool)

        start = self.k - len(items)

        for i, (hist_target, feat, hist_ts) in enumerate(
            items,
            start=start,
        ):
            raw[i] = feat
            age[i] = max(0.0, current_ts - hist_ts)
            same[i] = 1.0 if hist_target == target else 0.0
            mask[i] = True

        if mask.any():
            transformed = transform.transform(raw)
            transformed[~mask] = 0.0
        else:
            transformed = np.zeros(
                (
                    self.k,
                    len(BASE_FEATURES) + len(RECENCY_INDEXES),
                ),
                dtype=np.float32,
            )

        return transformed, age, same, mask


def build_online_history(
    changes: list[list[str]],
    end_exclusive: int,
    mu: float,
    history_k: int,
) -> tuple[TemporalState, OnlineHistory]:
    state = TemporalState(mu)
    hist = OnlineHistory(history_k)

    for i, commit in enumerate(changes[:end_exclusive]):
        ts = parse_timestamp(commit[0])
        files = commit[1:]
        hist.add_commit(state, files, i, ts)
        state.update(files, i, ts)

    return state, hist


def score_candidate(
    model: TemporalAttentionScorer,
    transform: FeatureTransform,
    history: OnlineHistory,
    state: TemporalState,
    source: str,
    target: str,
    commit_idx: int,
    ts: float,
    device: str,
) -> float:
    raw = state.features(
        source,
        target,
        commit_idx,
        ts,
    ).reshape(1, -1)

    cur = transform.transform(raw)
    hist_x, hist_age, same, mask = history.tensor_inputs(
        source,
        target,
        ts,
        transform,
    )

    with torch.no_grad():
        logits = model(
            torch.from_numpy(cur).to(device),
            torch.from_numpy(hist_x[None, ...]).to(device),
            torch.from_numpy(hist_age[None, ...]).to(device),
            torch.from_numpy(same[None, ...]).to(device),
            torch.from_numpy(mask[None, ...]).to(device),
        )

        return float(
            torch.sigmoid(logits)[0].cpu().item()
        )


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, float]:
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    f1 = safe_div(2 * ppv * sens, ppv + sens)

    mcc = safe_div(
        tp * tn - fp * fn,
        math.sqrt(
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
        ),
    )

    return {
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "gmean": math.sqrt(sens * spec),
        "f1": f1,
        "accuracy": safe_div(
            tp + tn,
            tp + tn + fp + fn,
        ),
        "mcc": mcc,
        "legacy_auc": 0.5 * (sens + spec),
    }


def evaluate_dfs(
    changes: list[list[str]],
    start: int,
    end: int,
    mu: float,
    predicted_size: int,
    threshold: float,
    model: TemporalAttentionScorer,
    transform: FeatureTransform,
    history_k: int,
    device: str,
    trace: bool = False,
) -> dict:
    state, history = build_online_history(
        changes,
        start,
        mu,
        history_k,
    )

    commit_rows = []
    candidate_rows = []

    for commit_idx in range(start, end):
        commit = changes[commit_idx]
        ts = parse_timestamp(commit[0])
        actual = list(dict.fromkeys(commit[1:]))
        actual_set = set(actual)

        seed = next(
            (f for f in actual if f in state.known),
            None,
        )

        if seed is None:
            history.add_commit(
                state,
                actual,
                commit_idx,
                ts,
            )
            state.update(
                actual,
                commit_idx,
                ts,
            )
            continue

        stack = [seed]
        predicted = [seed]
        predicted_set = {seed}
        depth = {seed: 0}
        calls = 0

        while stack and len(predicted) < predicted_size:
            source = stack.pop()

            for target in state.neighbors(source):
                if (
                    target in predicted_set
                    or len(predicted) >= predicted_size
                ):
                    continue

                prob = score_candidate(
                    model,
                    transform,
                    history,
                    state,
                    source,
                    target,
                    commit_idx,
                    ts,
                    device,
                )

                calls += 1
                accepted = prob > threshold

                if trace:
                    candidate_rows.append(
                        {
                            "commit_idx": commit_idx,
                            "seed": seed,
                            "source": source,
                            "target": target,
                            "dfs_depth": depth.get(source, 0),
                            "probability": prob,
                            "threshold": threshold,
                            "accepted": int(accepted),
                            "actual_target": int(
                                target in actual_set
                            ),
                            "cochange_prior": state.cochange(
                                source,
                                target,
                            ),
                        }
                    )

                if accepted:
                    predicted_set.add(target)
                    predicted.append(target)
                    depth[target] = (
                        depth.get(source, 0) + 1
                    )
                    stack.append(target)

        tp = len(actual_set & predicted_set)
        fn = len(actual_set - predicted_set)
        fp = len(predicted_set - actual_set)
        tn = max(
            0,
            len(state.known) - tp - fp - fn,
        )

        row = {
            "commit_idx": commit_idx,
            "seed": seed,
            "actual_size": len(actual_set),
            "predicted_size": len(predicted_set),
            "cap_hit": int(
                len(predicted_set) >= predicted_size
            ),
            "candidate_calls": calls,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

        row.update(
            metrics_from_counts(
                tp,
                tn,
                fp,
                fn,
            )
        )

        commit_rows.append(row)

        history.add_commit(
            state,
            actual,
            commit_idx,
            ts,
        )
        state.update(
            actual,
            commit_idx,
            ts,
        )

    if not commit_rows:
        return {
            "commit_rows": [],
            "candidate_rows": candidate_rows,
            "means": {},
            "behavior": {},
        }

    metric_keys = [
        "sensitivity",
        "specificity",
        "ppv",
        "gmean",
        "f1",
        "accuracy",
        "mcc",
        "legacy_auc",
    ]

    means = {
        k: float(
            np.mean(
                [r[k] for r in commit_rows]
            )
        )
        for k in metric_keys
    }

    cap = [
        r for r in commit_rows if r["cap_hit"]
    ]
    noncap = [
        r for r in commit_rows if not r["cap_hit"]
    ]

    behavior = {
        "evaluated_commits": len(commit_rows),
        "mean_actual_size": float(
            np.mean(
                [r["actual_size"] for r in commit_rows]
            )
        ),
        "mean_predicted_size": float(
            np.mean(
                [r["predicted_size"] for r in commit_rows]
            )
        ),
        "cap_hit_rate": float(
            np.mean(
                [r["cap_hit"] for r in commit_rows]
            )
        ),
        "mean_tp": float(
            np.mean(
                [r["tp"] for r in commit_rows]
            )
        ),
        "mean_fp": float(
            np.mean(
                [r["fp"] for r in commit_rows]
            )
        ),
        "mean_fp_cap_hit": (
            float(
                np.mean(
                    [r["fp"] for r in cap]
                )
            )
            if cap
            else None
        ),
        "mean_fp_noncap": (
            float(
                np.mean(
                    [r["fp"] for r in noncap]
                )
            )
            if noncap
            else None
        ),
        "candidate_calls": int(
            sum(
                r["candidate_calls"]
                for r in commit_rows
            )
        ),
    }

    return {
        "commit_rows": commit_rows,
        "candidate_rows": candidate_rows,
        "means": means,
        "behavior": behavior,
    }


def select_threshold(
    changes: list[list[str]],
    val_start: int,
    val_end: int,
    mu: float,
    predicted_size: int,
    model: TemporalAttentionScorer,
    transform: FeatureTransform,
    history_k: int,
    device: str,
    thresholds: Iterable[float],
) -> tuple[float, list[dict]]:
    rows = []
    best_t = None
    best_key = None

    for t in thresholds:
        result = evaluate_dfs(
            changes,
            val_start,
            val_end,
            mu,
            predicted_size,
            float(t),
            model,
            transform,
            history_k,
            device,
            trace=False,
        )

        m = result["means"]
        row = {
            "threshold": float(t),
            **m,
            **result["behavior"],
        }

        rows.append(row)

        key = (
            m.get("mcc", -1.0),
            m.get("f1", -1.0),
            float(t),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_t = float(t)

    if best_t is None:
        raise RuntimeError(
            "Threshold calibration produced no result"
        )

    return best_t, rows


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    obj: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
        ),
        encoding="utf-8",
    )
