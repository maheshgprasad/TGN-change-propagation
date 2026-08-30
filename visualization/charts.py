"""Altair charts for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import altair as alt

from data_loader import METRIC_LABELS, display_name
from metrics import format_metric

PRIMARY = "#1f4e79"
ACCENT = "#c0392b"
OK = "#1e8449"
WARN = "#d35400"


def metric_bar_chart(summary: pd.DataFrame) -> alt.Chart:
    frame = summary.copy()
    frame["label"] = frame["metric"].map(METRIC_LABELS)
    frame["display"] = [format_metric(m, v) for m, v in zip(frame["metric"], frame["mean"])]
    return (
        alt.Chart(frame)
        .mark_bar(color=PRIMARY)
        .encode(
            x=alt.X("mean:Q", title="Mean value (0–1 scale)", scale=alt.Scale(domain=[-0.05, 1.05])),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", "mean", "display"],
        )
        .properties(height=380)
    )


def runs_line_chart(runs: pd.DataFrame, selected: list[str]) -> alt.Chart:
    long = runs.melt(
        id_vars=["run"],
        value_vars=selected,
        var_name="metric",
        value_name="value",
    )
    long["label"] = long["metric"].map(METRIC_LABELS)
    return (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("run:O", title="Experimental run"),
            y=alt.Y("value:Q", title="Value", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("label:N", title="Metric"),
            tooltip=["run", "label", "value"],
        )
        .properties(height=360)
    )


def confusion_heatmap(tp: int, tn: int, fp: int, fn: int) -> alt.Chart:
    frame = pd.DataFrame(
        [
            {
                "predicted": "Predicted impacted",
                "actual": "Actual impacted",
                "label": "True Positive",
                "count": tp,
                "kind": "tp",
            },
            {
                "predicted": "Predicted impacted",
                "actual": "Actual not impacted",
                "label": "False Positive",
                "count": fp,
                "kind": "fp",
            },
            {
                "predicted": "Predicted not impacted",
                "actual": "Actual impacted",
                "label": "False Negative",
                "count": fn,
                "kind": "fn",
            },
            {
                "predicted": "Predicted not impacted",
                "actual": "Actual not impacted",
                "label": "True Negative",
                "count": tn,
                "kind": "tn",
            },
        ]
    )
    frame["text"] = frame["label"] + ": " + frame["count"].map(lambda n: f"{n:,}")
    return (
        alt.Chart(frame)
        .mark_rect(stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("actual:N", title="Actual"),
            y=alt.Y(
                "predicted:N",
                title="Predicted",
                sort=["Predicted impacted", "Predicted not impacted"],
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["tp", "tn", "fp", "fn"],
                    range=["#1e8449", "#7f8c8d", "#d35400", "#c0392b"],
                ),
                legend=None,
            ),
            tooltip=["label", "count"],
        )
        .properties(height=320, width=520)
    ) + (
        alt.Chart(frame)
        .mark_text(color="white", fontSize=14, fontWeight="bold")
        .encode(
            x="actual:N",
            y=alt.Y("predicted:N", sort=["Predicted impacted", "Predicted not impacted"]),
            text="text:N",
        )
    )


def stacked_effectiveness(detected: int, missed: int, correct: int, false_alarms: int) -> alt.Chart:
    frame = pd.DataFrame(
        [
            {"group": "Actually impacted files", "category": "Detected (TP)", "count": detected},
            {"group": "Actually impacted files", "category": "Missed (FN)", "count": missed},
            {"group": "Predicted impacted files", "category": "Correct warnings (TP)", "count": correct},
            {"group": "Predicted impacted files", "category": "False alarms (FP)", "count": false_alarms},
        ]
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("category:N", title=None),
            y=alt.Y("count:Q", title="Files"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=[
                        "Detected (TP)",
                        "Missed (FN)",
                        "Correct warnings (TP)",
                        "False alarms (FP)",
                    ],
                    range=[OK, ACCENT, PRIMARY, WARN],
                ),
                legend=None,
            ),
            tooltip=["category", "count"],
            facet=alt.Facet("group:N", title=None),
        )
        .properties(height=280)
    )


def project_comparison_chart(table: pd.DataFrame, metric: str) -> alt.Chart:
    frame = table.dropna(subset=[metric]).copy()
    frame["label"] = frame["project"].map(display_name)
    frame["display"] = [format_metric(metric, v) for v in frame[metric]]
    return (
        alt.Chart(frame)
        .mark_bar(color=PRIMARY)
        .encode(
            x=alt.X(f"{metric}:Q", title=METRIC_LABELS[metric], scale=alt.Scale(domain=[-0.05, 1.05])),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", metric, "display"],
        )
        .properties(height=max(220, 28 * len(frame) + 40), title=f"Mean {METRIC_LABELS[metric]} by project")
    )


def comparison_grouped_chart(table: pd.DataFrame) -> alt.Chart:
    long = table.melt(
        id_vars=["metric"],
        value_vars=["paper", "reproduction"],
        var_name="source",
        value_name="value",
    )
    long["label"] = long["metric"].map(METRIC_LABELS)
    long["source"] = long["source"].map(
        {"paper": "Published (paper)", "reproduction": "Reproduction mean"}
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title=None),
            y=alt.Y("value:Q", title="Value", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color("source:N", title=None),
            xOffset="source:N",
            tooltip=["label", "source", "value"],
        )
        .properties(height=340)
    )
