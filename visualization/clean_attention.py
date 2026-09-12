"""Visualization of the clean shared temporal-attention experiment.

This module reuses the existing Streamlit dashboard and reads only persisted
artifacts. It never trains a model or writes experiment outputs.
"""

from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import REPO_ROOT, RESULTS_ROOT, display_name, load_confusion_matrix

CLEAN_ROOT = REPO_ROOT / "Phase1CleanResults"
CONF_ROOT = RESULTS_ROOT / "ConfMatrix"

CLEAN_PROJECTS = (
    "alamofire",
    "ant",
    "cassandra",
    "laravel",
    "lucene",
    "monitorcontrol",
    "pydriller",
    "react",
    "rocketmqclients",
    "spark",
)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _row_metrics(tp: float, tn: float, fp: float, fn: float) -> dict[str, float]:
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * ppv * sensitivity, ppv + sensitivity)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "f1": f1,
        "mcc": mcc,
        "legacy_auc": 0.5 * (sensitivity + specificity),
    }


@st.cache_data(show_spinner=False)
def load_clean_summary(project: str, shuffle: int = 0) -> dict | None:
    path = CLEAN_ROOT / f"{project}_shuffle_{shuffle}" / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_clean_predictions(project: str, shuffle: int = 0) -> pd.DataFrame:
    path = CLEAN_ROOT / f"{project}_shuffle_{shuffle}" / "test_commit_predictions.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_candidate_trace(project: str, shuffle: int = 0) -> pd.DataFrame:
    path = CLEAN_ROOT / f"{project}_shuffle_{shuffle}" / "candidate_trace.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_threshold_grid(project: str, shuffle: int = 0) -> pd.DataFrame:
    path = CLEAN_ROOT / f"{project}_shuffle_{shuffle}" / "threshold_grid.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def baseline_shuffle_summary(project: str, shuffle: int, cap: int | None = None) -> dict | None:
    path = CONF_ROOT / f"directed_{project}_results_{shuffle}.csv"
    frame = load_confusion_matrix(path)
    if frame is None or frame.empty:
        return None

    per_row = [
        _row_metrics(float(r.tp), float(r.tn), float(r.fp), float(r.fn))
        for r in frame.itertuples(index=False)
    ]
    means = {
        key: float(np.mean([row[key] for row in per_row]))
        for key in per_row[0]
    }

    predicted_sizes = frame["tp"].to_numpy(dtype=float) + frame["fp"].to_numpy(dtype=float)
    fps = frame["fp"].to_numpy(dtype=float)
    behavior = {
        "rows": int(len(frame)),
        "mean_predicted_size": float(predicted_sizes.mean()),
        "mean_fp": float(fps.mean()),
    }

    if cap:
        hits = predicted_sizes >= cap
        behavior["cap_hit_rate"] = float(hits.mean())
        behavior["mean_fp_cap_hit"] = float(fps[hits].mean()) if hits.any() else None
        behavior["mean_fp_noncap"] = float(fps[~hits].mean()) if (~hits).any() else None

    return {"means": means, "behavior": behavior}


@st.cache_data(show_spinner=False)
def comparison_frame(shuffle: int = 0) -> pd.DataFrame:
    rows: list[dict] = []
    for project in CLEAN_PROJECTS:
        clean = load_clean_summary(project, shuffle)
        if clean is None:
            continue
        cap = int(clean["predicted_size_cap"])
        baseline = baseline_shuffle_summary(project, shuffle, cap)

        row = {
            "project": project,
            "Project": display_name(project),
            "threshold": float(clean["selected_threshold"]),
            "runtime_seconds": float(clean["timing_seconds"]["total"]),
            "evaluated_commits": int(clean["behavior"]["evaluated_commits"]),
            "clean_f1": float(clean["test_means"]["f1"]),
            "clean_mcc": float(clean["test_means"]["mcc"]),
            "clean_recall": float(clean["test_means"]["sensitivity"]),
            "clean_ppv": float(clean["test_means"]["ppv"]),
            "clean_mean_fp": float(clean["behavior"]["mean_fp"]),
            "clean_mean_predicted_size": float(clean["behavior"]["mean_predicted_size"]),
            "clean_cap_hit_rate": float(clean["behavior"]["cap_hit_rate"]),
        }

        if baseline:
            row.update(
                {
                    "baseline_f1": baseline["means"]["f1"],
                    "baseline_mcc": baseline["means"]["mcc"],
                    "baseline_recall": baseline["means"]["sensitivity"],
                    "baseline_ppv": baseline["means"]["ppv"],
                    "baseline_mean_fp": baseline["behavior"]["mean_fp"],
                    "baseline_mean_predicted_size": baseline["behavior"]["mean_predicted_size"],
                    "baseline_cap_hit_rate": baseline["behavior"].get("cap_hit_rate"),
                }
            )
            row["delta_f1"] = row["clean_f1"] - row["baseline_f1"]
            row["delta_mcc"] = row["clean_mcc"] - row["baseline_mcc"]
            row["delta_recall"] = row["clean_recall"] - row["baseline_recall"]
            row["delta_ppv"] = row["clean_ppv"] - row["baseline_ppv"]
            row["delta_mean_fp"] = row["clean_mean_fp"] - row["baseline_mean_fp"]
        else:
            for key in (
                "baseline_f1",
                "baseline_mcc",
                "baseline_recall",
                "baseline_ppv",
                "baseline_mean_fp",
                "baseline_mean_predicted_size",
                "baseline_cap_hit_rate",
                "delta_f1",
                "delta_mcc",
                "delta_recall",
                "delta_ppv",
                "delta_mean_fp",
            ):
                row[key] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _comparison_chart(frame: pd.DataFrame, metric: str, title: str) -> alt.Chart:
    base_col = f"baseline_{metric}"
    clean_col = f"clean_{metric}"
    plot = frame.dropna(subset=[base_col, clean_col]).copy()
    long = plot.melt(
        id_vars=["Project"],
        value_vars=[base_col, clean_col],
        var_name="Model",
        value_name="Score",
    )
    long["Model"] = long["Model"].map(
        {base_col: "Germanos baseline", clean_col: "Clean attention"}
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Project:N", sort="-x", title=None),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title=title),
            color=alt.Color("Model:N", title=None),
            yOffset="Model:N",
            tooltip=["Project", "Model", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(height=max(280, len(plot) * 34))
    )


def _delta_chart(frame: pd.DataFrame) -> alt.Chart:
    plot = frame.dropna(subset=["delta_f1", "delta_mcc"]).copy()
    long = plot.melt(
        id_vars=["Project"],
        value_vars=["delta_f1", "delta_mcc"],
        var_name="Metric",
        value_name="Delta",
    )
    long["Metric"] = long["Metric"].map({"delta_f1": "Δ F1", "delta_mcc": "Δ MCC"})
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            y=alt.Y("Project:N", title=None),
            x=alt.X("Delta:Q", title="Clean attention − baseline"),
            color=alt.Color("Metric:N", title=None),
            yOffset="Metric:N",
            tooltip=["Project", "Metric", alt.Tooltip("Delta:Q", format="+.3f")],
        )
        .properties(height=max(280, len(plot) * 34))
    )


def _behavior_chart(baseline: dict, clean: dict) -> alt.Chart:
    data = pd.DataFrame(
        [
            {
                "Model": "Germanos baseline",
                "Mean FP": baseline["behavior"]["mean_fp"],
                "Mean predicted size": baseline["behavior"]["mean_predicted_size"],
            },
            {
                "Model": "Clean attention",
                "Mean FP": clean["behavior"]["mean_fp"],
                "Mean predicted size": clean["behavior"]["mean_predicted_size"],
            },
        ]
    )
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title=None),
            y=alt.Y("Mean FP:Q", title="Mean false positives per evaluated commit"),
            tooltip=[
                "Model",
                alt.Tooltip("Mean FP:Q", format=".3f"),
                alt.Tooltip("Mean predicted size:Q", format=".3f"),
            ],
        )
        .properties(height=300)
    )


def _evidence_text(row: pd.Series) -> str:
    if pd.isna(row.get("baseline_f1")):
        return (
            "No persisted shuffle-0 baseline confusion file is available for this project, "
            "so it is shown as clean-model feasibility evidence only."
        )

    df1 = float(row["delta_f1"])
    drec = float(row["delta_recall"])
    dppv = float(row["delta_ppv"])
    dfp = float(row["delta_mean_fp"])

    if df1 > 0.005 and dfp < -0.05:
        return (
            f"Descriptive evidence: mean false positives fell by {abs(dfp):.2f} per evaluated "
            f"commit. Precision changed by {dppv:+.3f}, and F1 improved by {df1:+.3f}."
        )
    if df1 > 0.005 and drec > 0.01:
        return (
            f"Descriptive evidence: recall increased by {drec:+.3f}. The additional recovery "
            f"was large enough to raise F1 by {df1:+.3f}, despite a precision change of {dppv:+.3f}."
        )
    if df1 < -0.005 and dfp > 0.05:
        return (
            f"Descriptive evidence: mean false positives increased by {dfp:.2f} per evaluated "
            f"commit. Recall changed by {drec:+.3f}, but F1 fell by {abs(df1):.3f}."
        )
    return (
        f"The trade-off is small: recall changed by {drec:+.3f}, precision by {dppv:+.3f}, "
        f"and F1 by {df1:+.3f}. Treat this as near-neutral rather than a clear win or loss."
    )


def _render_candidate_explorer(project: str, predictions: pd.DataFrame, trace: pd.DataFrame) -> None:
    st.subheader("Commit-level evidence explorer")
    if predictions.empty:
        st.info("No clean test prediction trace is available for this project.")
        return

    commits = predictions["commit_idx"].astype(int).tolist()
    commit_idx = st.selectbox(
        "Evaluated test commit",
        options=commits,
        key=f"clean_commit_{project}",
    )
    row = predictions[predictions["commit_idx"] == commit_idx].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Seed", str(row["seed"]))
    c2.metric("Actual size", int(row["actual_size"]))
    c3.metric("Predicted size", int(row["predicted_size"]))
    c4.metric("TP / FP / FN", f"{int(row['tp'])} / {int(row['fp'])} / {int(row['fn'])}")
    c5.metric("Candidate calls", int(row["candidate_calls"]))

    if int(row["cap_hit"]):
        st.warning("This prediction reached the project prediction-size cap.")
    else:
        st.caption("This prediction did not reach the project prediction-size cap.")

    subset = trace[trace["commit_idx"] == commit_idx].copy() if not trace.empty else pd.DataFrame()
    if subset.empty:
        st.info("No candidate-level calls were recorded for this evaluated commit.")
        return

    def outcome(r):
        if int(r["accepted"]) and int(r["actual_target"]):
            return "Accepted + actual (TP candidate)"
        if int(r["accepted"]) and not int(r["actual_target"]):
            return "Accepted + not actual (FP candidate)"
        if not int(r["accepted"]) and int(r["actual_target"]):
            return "Rejected + actual (missed candidate)"
        return "Rejected + not actual"

    subset["Outcome"] = subset.apply(outcome, axis=1)
    show = subset[
        [
            "source",
            "target",
            "dfs_depth",
            "probability",
            "threshold",
            "cochange_prior",
            "Outcome",
        ]
    ].copy()
    show.columns = [
        "Source",
        "Target",
        "DFS depth",
        "Probability",
        "Threshold",
        "Prior co-change",
        "Decision outcome",
    ]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.caption(
        "This is a decision trace, not a causal explanation of attention weights. "
        "It shows which candidates were considered, their score, threshold decision, "
        "prior co-change evidence, and whether the target was actually changed."
    )


def render_clean_attention(preferred_project: str | None = None) -> None:
    st.header("Baseline vs clean temporal attention")
    st.markdown(
        "This section uses the existing dashboard to compare the frozen Germanos shuffle-0 "
        "baseline with the clean shared temporal-attention implementation. It also exposes "
        "the persisted prediction traces used to explain why a repository improved or worsened."
    )

    frame = comparison_frame(0)
    if frame.empty:
        st.warning("No clean-attention results were found under Phase1CleanResults.")
        return

    comparable = frame.dropna(subset=["delta_f1", "delta_mcc"])
    f1_wins = int((comparable["delta_f1"] > 0).sum())
    mcc_wins = int((comparable["delta_mcc"] > 0).sum())

    a, b, c, d = st.columns(4)
    a.metric("Clean projects", len(frame))
    b.metric("Directly comparable", len(comparable))
    c.metric("F1 wins", f"{f1_wins} / {len(comparable)}")
    d.metric("MCC wins", f"{mcc_wins} / {len(comparable)}")

    table = frame[
        [
            "Project",
            "baseline_f1",
            "clean_f1",
            "delta_f1",
            "baseline_mcc",
            "clean_mcc",
            "delta_mcc",
        ]
    ].copy()
    table.columns = [
        "Project",
        "Baseline F1",
        "Clean F1",
        "Δ F1",
        "Baseline MCC",
        "Clean MCC",
        "Δ MCC",
    ]
    display_table = table.copy()
    for column in ("Baseline F1", "Clean F1", "Baseline MCC", "Clean MCC"):
        display_table[column] = display_table[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    for column in ("Δ F1", "Δ MCC"):
        display_table[column] = display_table[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:+.3f}"
        )

    st.dataframe(
        display_table,
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Ant has no persisted shuffle-0 baseline confusion file, so no direct delta is reported for Ant."
    )

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            _comparison_chart(frame, "f1", "F1"),
            use_container_width=True,
        )
    with right:
        st.altair_chart(_delta_chart(frame), use_container_width=True)

    project_options = frame["project"].tolist()
    default = preferred_project if preferred_project in project_options else project_options[0]
    project = st.selectbox(
        "Evidence project",
        options=project_options,
        index=project_options.index(default),
        format_func=display_name,
        key="clean_evidence_project",
    )

    clean = load_clean_summary(project, 0)
    assert clean is not None
    cap = int(clean["predicted_size_cap"])
    baseline = baseline_shuffle_summary(project, 0, cap)
    row = frame[frame["project"] == project].iloc[0]

    st.subheader(f"Why did {display_name(project)} change?")
    st.info(_evidence_text(row))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected threshold", f"{float(clean['selected_threshold']):.2f}")
    c2.metric("Prediction cap", cap)
    c3.metric("Evaluated commits", int(clean["behavior"]["evaluated_commits"]))
    c4.metric("Total clean runtime", f"{float(clean['timing_seconds']['total']):.1f} s")

    if baseline is not None:
        x1, x2, x3 = st.columns(3)
        x1.metric(
            "Mean false positives",
            f"{clean['behavior']['mean_fp']:.3f}",
            delta=f"{clean['behavior']['mean_fp'] - baseline['behavior']['mean_fp']:+.3f}",
            delta_color="inverse",
        )
        x2.metric(
            "Mean predicted size",
            f"{clean['behavior']['mean_predicted_size']:.3f}",
            delta=f"{clean['behavior']['mean_predicted_size'] - baseline['behavior']['mean_predicted_size']:+.3f}",
            delta_color="off",
        )
        x3.metric(
            "Cap-hit rate",
            f"{clean['behavior']['cap_hit_rate'] * 100:.1f}%",
            delta=f"{(clean['behavior']['cap_hit_rate'] - baseline['behavior'].get('cap_hit_rate', 0.0)) * 100:+.1f} pp",
            delta_color="off",
        )
        st.altair_chart(_behavior_chart(baseline, clean), use_container_width=True)
    else:
        st.warning("Direct baseline behavior comparison is unavailable for this project.")

    with st.expander("Threshold calibration evidence"):
        grid = load_threshold_grid(project, 0)
        if grid.empty:
            st.info("No threshold grid is available.")
        else:
            long = grid.melt(
                id_vars=["threshold"],
                value_vars=["f1", "mcc"],
                var_name="Metric",
                value_name="Score",
            )
            long["Metric"] = long["Metric"].map({"f1": "F1", "mcc": "MCC"})
            st.altair_chart(
                alt.Chart(long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("threshold:Q", title="Decision threshold"),
                    y=alt.Y("Score:Q", title="Validation score"),
                    color=alt.Color("Metric:N", title=None),
                    tooltip=[
                        alt.Tooltip("threshold:Q", format=".2f"),
                        "Metric",
                        alt.Tooltip("Score:Q", format=".3f"),
                    ],
                )
                .properties(height=300),
                use_container_width=True,
            )
            st.caption(
                "Threshold selection uses validation DFS replay. Final test rows are not used to choose the threshold."
            )

    _render_candidate_explorer(
        project,
        load_clean_predictions(project, 0),
        load_candidate_trace(project, 0),
    )
