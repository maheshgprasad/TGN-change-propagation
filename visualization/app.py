"""Streamlit dashboard for TGN change-propagation result CSVs.

Read-only: never writes to Results/, ShuffledData/, or author source files.

Run from the repository root:
    streamlit run visualization/app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from charts import (
    comparison_grouped_chart,
    confusion_heatmap,
    metric_bar_chart,
    project_comparison_chart,
    runs_line_chart,
    stacked_effectiveness,
)
from data_loader import (
    DATA_ROOT,
    METRIC_KEYS,
    METRIC_LABELS,
    RESULTS_ROOT,
    ProjectResults,
    discover_projects,
    display_name,
    filter_changesets_like_authors,
    list_changeset_projects,
    load_all_projects,
    load_raw_changesets,
)
from graph_view import (
    build_subgraph,
    file_label,
    graph_html,
    history_up_to,
    neighbor_rows,
)
from metrics import (
    aggregate_confusion,
    comparison_table,
    detection_breakdown,
    first_hyperparameters,
    format_metric,
    project_means_table,
    runs_as_frame,
    summarize_runs,
)
from clean_attention import render_clean_attention

PAGES = (
    "Understand the task",
    "This project's numbers",
    "Compare",
    "Clean attention evidence",
    "Input graph",
)
ASSETS = Path(__file__).resolve().parent / "assets"
DIAGRAM = ASSETS / "prediction_diagram.svg"
COCHANGE_LEGEND = ASSETS / "cochange_legend.svg"

st.set_page_config(
    page_title="TGN Change Propagation — Results",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {background:#f7f9fc;border:1px solid #d6e0ea;border-radius:10px;padding:1rem 1.1rem;}
    .metric-card h3 {margin:0;font-size:0.85rem;color:#5d6d7e;font-weight:600;}
    .metric-card p {margin:0.25rem 0 0;font-size:1.8rem;font-weight:700;color:#1f4e79;}
    .fn-box {background:#fdedec;border-left:4px solid #c0392b;padding:0.8rem 1rem;margin:0.4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_projects():
    return load_all_projects()


def render_card(title: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><h3>{title}</h3><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


def takeaway_for_project(project: ProjectResults) -> dict:
    """Headline numbers for the selected project (means across runs)."""
    summary = summarize_runs(project)
    means = {row["metric"]: row["mean"] for _, row in summary.iterrows()}
    n_runs = int(summary["n"].max()) if not summary.empty else 0
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for frame in project.confusion_runs.values():
        part = aggregate_confusion(frame)
        for key in totals:
            totals[key] += part[key]
    actual = totals["tp"] + totals["fn"]
    recall = means.get("sensitivity")
    missed_share = None
    if actual:
        missed_share = totals["fn"] / actual
    elif recall is not None:
        missed_share = 1.0 - float(recall)
    return {
        "means": means,
        "n_runs": n_runs,
        "counts": totals,
        "actual_impacted": actual,
        "missed_share": missed_share,
        "has_confusion": bool(project.confusion_runs),
    }


def render_takeaway(result: ProjectResults | None) -> None:
    if result is None or result.n_metric_runs == 0:
        st.info("Select a project with metrics CSV rows to see a one-line summary of how the model did.")
        return
    story = takeaway_for_project(result)
    means = story["means"]
    recall = means.get("sensitivity")
    precision = means.get("ppv")
    auc = means.get("auc")
    missed = story["missed_share"]
    name = display_name(result.project)
    st.subheader("What this project shows")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_card("Recall (found)", format_metric("sensitivity", recall))
    with c2:
        render_card("Impacted files missed", format_metric("sensitivity", missed))
    with c3:
        render_card("AUC", format_metric("auc", auc))

    bits = [
        f"On **{name}**, across **{story['n_runs']}** experimental run(s), "
        f"the model found **{format_metric('sensitivity', recall)}** of actually impacted files."
    ]
    if story["has_confusion"] and story["actual_impacted"]:
        bits.append(
            f"Confusion-matrix files record **{story['counts']['fn']:,}** misses out of "
            f"**{story['actual_impacted']:,}** actually impacted files (all shuffles combined)."
        )
    elif missed is not None:
        bits.append(
            f"That means about **{format_metric('sensitivity', missed)}** of impacted files were false negatives."
        )
    bits.append(
        f"When it warned, precision was **{format_metric('ppv', precision)}**. "
        f"AUC is **{format_metric('auc', auc)}** — ranking is useful, but missed impacts remain "
        "the main risk for change-impact analysis."
    )
    st.markdown(" ".join(bits))
    st.caption("Headline figures are means across runs. Open a section in the sidebar for the supporting charts.")


def render_understand_task() -> None:
    st.header("What the model predicts")
    st.markdown(
        "The model receives a **changed source file** and scores every other file for "
        "likely co-change. Each scored file lands in one of the four outcomes below."
    )
    left, right = st.columns([1.55, 1])
    with left:
        st.image(str(DIAGRAM), use_container_width=True)
        st.caption(
            "Example: `A.py` changed. `B.py` is a correct impact warning (TP); "
            "`C.py` was actually impacted but missed (FN); `D.py` is a false alarm (FP); "
            "`E.py` was correctly left alone (TN)."
        )
    with right:
        st.markdown(
            """
            - **True Positive (TP):** impacted file correctly predicted as impacted
            - **True Negative (TN):** non-impacted file correctly left alone
            - **False Positive (FP):** non-impacted file incorrectly flagged
            - **False Negative (FN):** impacted file missed — the model did not warn
            """
        )
        st.markdown(
            '<div class="fn-box"><b>False Negative:</b> a file was actually impacted, '
            "but the model missed it. This is particularly important for change-impact analysis.</div>",
            unsafe_allow_html=True,
        )

    st.header("How to read the metrics")
    st.caption("Open a term only when you need it. Headline numbers live under This project's numbers.")
    glossary = (
        (
            "Sensitivity / Recall",
            "Of all files that were actually impacted, how many did the model successfully find?",
        ),
        (
            "PPV / Precision",
            "When the model predicts that a file is impacted, how often is that prediction correct?",
        ),
        (
            "Specificity",
            "Of the files that were not impacted, how many did the model correctly leave alone?",
        ),
        ("F1", "Balances Precision and Recall."),
        (
            "AUC",
            "Measures how well the model distinguishes impacted files from non-impacted files.",
        ),
        (
            "G-Mean",
            "Geometric mean of Recall and Specificity — useful when classes are imbalanced.",
        ),
        (
            "MCC",
            "Matthews correlation: a balanced score that uses all four confusion-matrix cells. "
            "Shown on a -1 to 1 scale, not as a percentage.",
        ),
        (
            "Accuracy",
            "Share of all files (impacted and not) that were classified correctly. "
            "This can look high when most files are true negatives.",
        ),
    )
    for title, body in glossary:
        with st.expander(title):
            st.write(body)


def render_project_numbers(result: ProjectResults | None) -> None:
    st.header("Experiment overview")
    if result is None:
        st.info("Select a project after result files are generated.")
    else:
        cols = st.columns(4)
        cols[0].metric("Project", display_name(result.project))
        cols[1].metric("Experiment runs", str(result.n_metric_runs))
        cols[2].metric(
            "Graph representation",
            result.graph_type.title() if result.graph_type else "Directed",
        )
        cols[3].metric(
            "Confusion-matrix runs",
            ", ".join(str(i) for i in result.confusion_run_ids) or "None",
        )
        hypers = first_hyperparameters(result)
        if hypers:
            st.caption(
                "Hyperparameters recorded with the metrics: "
                + ", ".join(f"{k}={v}" for k, v in hypers.items())
            )
        for message in result.warnings:
            st.caption(f"Note: {message}")

    st.header("Performance metrics")
    summary = pd.DataFrame()
    if result is None or result.n_metric_runs == 0:
        st.info("No metrics CSV rows are available for this project yet.")
    else:
        summary = summarize_runs(result)
        means = {row["metric"]: row["mean"] for _, row in summary.iterrows()}
        st.subheader(display_name(result.project))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_card("Recall", format_metric("sensitivity", means.get("sensitivity")))
        with c2:
            render_card("Precision", format_metric("ppv", means.get("ppv")))
        with c3:
            render_card("F1", format_metric("fmeasure", means.get("fmeasure")))
        with c4:
            render_card("AUC", format_metric("auc", means.get("auc")))

        st.markdown("#### Mean, variation, and range across runs")
        display = summary.copy()
        display["Metric"] = display["metric"].map(METRIC_LABELS)
        display["Mean"] = [format_metric(m, v) for m, v in zip(display["metric"], display["mean"])]
        display["Std. dev."] = [format_metric(m, v) for m, v in zip(display["metric"], display["std"])]
        display["Minimum"] = [format_metric(m, v) for m, v in zip(display["metric"], display["min"])]
        display["Maximum"] = [format_metric(m, v) for m, v in zip(display["metric"], display["max"])]
        st.dataframe(
            display[["Metric", "Mean", "Std. dev.", "Minimum", "Maximum", "n"]].rename(
                columns={"n": "Runs"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Underlying values are unchanged; percentages are display-only.")

    st.header("Metrics across runs")
    if result is None or result.n_metric_runs == 0:
        st.info("Need at least one metrics row to plot run-to-run variation.")
    else:
        runs = runs_as_frame(result)
        default_metrics = [m for m in ("sensitivity", "ppv", "fmeasure", "auc") if m in runs.columns]
        selected = st.multiselect(
            "Metrics to plot",
            options=list(METRIC_KEYS),
            default=default_metrics,
            format_func=lambda key: METRIC_LABELS[key],
        )
        if selected:
            st.altair_chart(runs_line_chart(runs, selected), use_container_width=True)
        else:
            st.caption("Select one or more metrics.")

    st.header("Mean metric comparison")
    if summary.empty:
        st.info("No means to plot yet.")
    else:
        st.altair_chart(metric_bar_chart(summary), use_container_width=True)

    st.header("Confusion matrix")
    counts = None
    if result is None or not result.confusion_run_ids:
        st.info("No confusion-matrix CSV files were found for this project.")
    else:
        run_id = st.selectbox(
            "Confusion-matrix run (shuffle index in the filename)",
            options=result.confusion_run_ids,
            format_func=lambda i: f"Run / shuffle {i}",
        )
        cm = result.confusion_runs[run_id]
        st.caption(
            f"{len(cm)} test change-set rows in `directed_{result.project}_results_{run_id}.csv`. "
            "Each row is one test change set: TP, TN, FP, FN."
        )
        mode = st.radio(
            "Display",
            ["Aggregate all test change-sets in this run", "Single test change-set"],
            horizontal=True,
        )
        if mode.startswith("Single"):
            row_index = st.slider("Change-set row", 0, len(cm) - 1, 0)
            row = cm.iloc[row_index]
            counts = {
                "tp": int(row["tp"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            }
        else:
            counts = aggregate_confusion(cm)

        st.altair_chart(
            confusion_heatmap(counts["tp"], counts["tn"], counts["fp"], counts["fn"]),
            use_container_width=True,
        )
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**True Positive:** Model correctly identified an impacted file.")
            st.markdown("**True Negative:** Model correctly left a non-impacted file alone.")
        with t2:
            st.markdown("**False Positive:** Model warned about a file that was not impacted.")
            st.markdown(
                '<div class="fn-box"><b>False Negative:</b> Model missed a file that was actually impacted.</div>',
                unsafe_allow_html=True,
            )

    st.header("Prediction effectiveness")
    if not counts:
        st.info("Open a confusion-matrix run above to see detection vs miss counts.")
    else:
        parts = detection_breakdown(counts)
        st.write(
            f"**Actual impacted files:** {parts['actual_impacted']:,}  ·  "
            f"**Predicted impacted files:** {parts['predicted_impacted']:,}"
        )
        st.altair_chart(
            stacked_effectiveness(
                parts["detected"],
                parts["missed"],
                parts["correct_warnings"],
                parts["false_alarms"],
            ),
            use_container_width=True,
        )
        if parts["actual_impacted"]:
            recall = parts["detected"] / parts["actual_impacted"]
            st.write(
                f"Recall view: found **{format_metric('sensitivity', recall)}** of actually impacted files "
                f"({parts['missed']:,} missed)."
            )
        if parts["predicted_impacted"]:
            precision = parts["correct_warnings"] / parts["predicted_impacted"]
            st.write(
                f"Precision view: **{format_metric('ppv', precision)}** of impact warnings were correct "
                f"({parts['false_alarms']:,} false alarms)."
            )


def render_compare(result: ProjectResults | None, projects: dict) -> None:
    st.header("Project comparison")
    means_table = project_means_table(projects)
    if means_table.empty:
        st.info("Need metrics for at least one project to compare.")
    else:
        compare_metric = st.selectbox(
            "Metric for comparison",
            options=list(METRIC_KEYS),
            index=list(METRIC_KEYS).index("auc"),
            format_func=lambda key: METRIC_LABELS[key],
            key="compare_metric",
        )
        st.altair_chart(
            project_comparison_chart(means_table, compare_metric),
            use_container_width=True,
        )
        pretty = means_table.copy()
        pretty.insert(0, "Project", pretty["project"].map(display_name))
        for key in METRIC_KEYS:
            if key in pretty.columns:
                pretty[METRIC_LABELS[key]] = [format_metric(key, v) for v in pretty[key]]
        show_cols = ["Project"] + [
            METRIC_LABELS[k] for k in METRIC_KEYS if METRIC_LABELS[k] in pretty.columns
        ]
        st.dataframe(pretty[show_cols], hide_index=True, use_container_width=True)

    st.header("Published vs reproduced results")
    st.caption(
        "Optional. Fill `visualization/reference_results.csv` with values from the paper. "
        "Empty or unmatched project names are skipped; numbers are not invented here."
    )
    if result is None:
        st.info("Select a project with reproduction metrics to compare.")
        return
    cmp = comparison_table(result)
    if cmp.empty:
        st.info(
            f"No published reference row for **{display_name(result.project)}**. "
            "Add a row to `visualization/reference_results.csv` when you have the paper numbers."
        )
        return
    view = cmp.copy()
    view["Metric"] = view["metric"].map(METRIC_LABELS)
    view["Paper"] = [format_metric(m, v) for m, v in zip(view["metric"], view["paper"])]
    view["Reproduction"] = [format_metric(m, v) for m, v in zip(view["metric"], view["reproduction"])]
    view["Absolute difference"] = [
        format_metric(m, v) for m, v in zip(view["metric"], view["absolute_difference"])
    ]
    st.dataframe(
        view[["Metric", "Paper", "Reproduction", "Absolute difference"]],
        hide_index=True,
        use_container_width=True,
    )
    st.altair_chart(comparison_grouped_chart(cmp), use_container_width=True)
    st.caption("Differences are shown without judging whether the reproduction succeeded.")


def render_graph(project_id: str | None) -> None:
    st.header("Co-change graph")
    st.markdown(
        "This is the **historical co-change graph the TGN reads as input**, not the "
        "model's predicted change-set. Each **node is a file** (numeric IDs from the "
        "change-set CSV — the authors never stored paths). Each **arrow A -> B** is the "
        "share of file A's past commits that also included file B."
    )
    st.image(str(COCHANGE_LEGEND), use_container_width=True)

    graph_projects = list_changeset_projects()
    if not graph_projects:
        st.warning(
            "Graph visualization unavailable from the current result artifacts. "
            "Additional graph-state export is required, and `ShuffledData` change-sets were not found."
        )
        return

    default_graph = project_id if project_id in graph_projects else graph_projects[0]
    g_project = st.selectbox(
        "Project",
        options=graph_projects,
        index=graph_projects.index(default_graph),
        format_func=display_name,
        key="graph_project",
        help="Which software project's change-sets to reconstruct.",
    )
    shuffle_options = [
        i for i in range(5) if (DATA_ROOT / str(i) / "ChangeSets" / f"{g_project}.csv").is_file()
    ]
    g_shuffle = st.selectbox(
        "Shuffle / fold",
        shuffle_options,
        format_func=lambda i: f"Shuffle {i}",
        help="The authors repeated the experiment on shuffled splits. Each shuffle is a different ordering of the same history.",
    )

    raw = load_raw_changesets(g_project, g_shuffle)
    if not raw:
        st.warning("Could not read that change-set file.")
        return
    filtered = filter_changesets_like_authors(raw)
    if not filtered:
        st.warning("After the authors' commit-size filter, no commits remain.")
        return

    commit_idx = st.slider(
        "History through commit",
        0,
        len(filtered) - 1,
        len(filtered) - 1,
        help="Only commits from the start of this filtered history through this index are used. The last commit shows the fullest graph.",
    )
    min_w = st.slider(
        "Hide weak links (minimum score)",
        0.0,
        1.0,
        0.10,
        0.01,
        help="0.10 means: only show B if it appeared in at least 10% of the source file's commits. Raise this to declutter.",
    )
    max_nodes = st.slider(
        "Maximum companion files",
        5,
        40,
        12,
        help="Caps how many files are drawn so the picture stays readable.",
    )
    history = history_up_to(g_project, g_shuffle, commit_idx)
    if history is None:
        st.warning(
            "Graph visualization unavailable from the current result artifacts. "
            "Additional graph-state export is required."
        )
        return

    changes, changes_of_file, changes_set = history
    files = sorted(changes_of_file, key=lambda name: len(changes_of_file[name]), reverse=True)
    OVERVIEW = "Overview: busiest files (dense cluster)"
    source = st.selectbox(
        "Ask: if this file changed, which files historically changed with it?",
        options=files[:500] + [OVERVIEW],
        format_func=lambda name: (
            OVERVIEW
            if name == OVERVIEW
            else f"{file_label(name)}  ·  {len(changes_of_file[name])} commits in this slice"
        ),
        help="Default is the file that changed most often — the clearest ego-network. Overview draws a cluster of busy files and is harder to read.",
    )
    source_file = None if source == OVERVIEW else source
    with st.spinner("Reconstructing co-change edges…"):
        graph = build_subgraph(
            changes_of_file,
            changes_set,
            commit_idx,
            min_w,
            max_nodes,
            source_file,
        )
    commit = changes[commit_idx]
    commit_stamp = commit[0] if commit else ""
    commit_files = commit[1:] if len(commit) > 1 else []
    n1, n2, n3 = st.columns(3)
    n1.metric("Files drawn", str(graph.number_of_nodes()))
    n2.metric("Arrows drawn", str(graph.number_of_edges()))
    n3.metric("History slice", f"{commit_idx + 1} / {len(filtered)}")
    if source_file:
        n_src = len(changes_of_file.get(source_file, []))
        st.info(
            f"**{file_label(source_file)}** appears in **{n_src}** commits through this slice. "
            "Orange nodes are the strongest historical companions. "
            "Larger nodes changed more often."
        )
    else:
        st.info(
            "Overview mode: busiest files and every link among them that passes the score filter. "
            "Prefer a single source file if this looks crowded."
        )
    st.caption(
        f"{display_name(g_project)} · shuffle {g_shuffle} · "
        f"commit {commit_idx + 1} timestamp `{commit_stamp}`"
    )
    if commit_files:
        shown = ", ".join(file_label(fid) for fid in commit_files[:24])
        extra = "" if len(commit_files) <= 24 else f" … (+{len(commit_files) - 24} more)"
        st.caption(f"Files in this commit: {shown}{extra}")

    vis, table = st.columns([1.35, 1])
    with vis:
        st.markdown(
            '<p style="margin:0 0 0.4rem;font-size:0.9rem;">'
            '<span style="color:#c0392b;">●</span> Source file &nbsp;&nbsp;'
            '<span style="color:#d35400;">●</span> Historically co-changed &nbsp;&nbsp;'
            "Hover a node or arrow for the sentence-form score."
            "</p>",
            unsafe_allow_html=True,
        )
        st.components.v1.html(
            graph_html(
                graph,
                source_file,
                {name: len(times) for name, times in changes_of_file.items()},
            ),
            height=580,
            scrolling=False,
        )
    with table:
        if source_file:
            rows = neighbor_rows(graph, source_file, changes_of_file)
            st.markdown(f"**Ranked companions of {file_label(source_file)}**")
            if not rows:
                st.info(
                    "No companion files pass the current minimum score. "
                    'Lower "Hide weak links" or pick a busier source file.'
                )
            else:
                frame = pd.DataFrame(rows)
                st.dataframe(
                    frame[
                        [
                            "Rank",
                            "File",
                            "When the source changed, this file was in the same commit",
                            "Times this file changed in this history slice",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                top = rows[0]
                st.caption(
                    f"Strongest link: when {file_label(source_file)} changed, "
                    f"{top['File']} was in the same commit "
                    f"{top['When the source changed, this file was in the same commit']} of the time."
                )
        elif graph.number_of_edges() == 0:
            st.info("No edges pass the current minimum score filter.")
        else:
            st.caption("Select a single source file to see a ranked companion list.")

    with st.expander("What this view is — and is not"):
        st.markdown(
            "The authors never write NetworkX graphs, adjacency matrices, or edge lists. "
            "The TGN's co-change graph (`temporal_node_cochanges`) exists only in memory during training. "
            "This panel **rebuilds** directed co-changeability from `ShuffledData` change-sets with the "
            "same formula as `TGN_model.py`. It is an input-graph sketch, not a saved model snapshot "
            "and not the predicted change-set from Understand the task."
        )


st.title("TGN Change Propagation — Result Dashboard")
st.caption(
    "Visualization layer for the authors' Temporal Graph Network experiments. "
    "This app only reads existing CSV outputs and change-set files."
)

projects = cached_projects()
available = discover_projects()

if not available:
    st.warning(
        "No result CSV files were found under `Results/Metrics` or `Results/ConfMatrix`. "
        "Run the authors' experiment first, then refresh this page. "
        "Change-set graph reconstruction is still available below if `ShuffledData` is present."
    )

project_id = None
result = None
if available:
    labels = {name: display_name(name) for name in available}
    project_id = st.sidebar.selectbox(
        "Project",
        options=available,
        format_func=lambda name: labels[name],
    )
    result = projects[project_id]

page = st.sidebar.radio("Go to", list(PAGES), index=1)
st.sidebar.caption("Only the selected section is built, so the graph does not run until you open it.")

st.sidebar.markdown("### Data locations")
st.sidebar.code(str(RESULTS_ROOT), language="text")
st.sidebar.caption("Files are opened read-only.")

render_takeaway(result)

if page == "Understand the task":
    render_understand_task()
elif page == "This project's numbers":
    render_project_numbers(result)
elif page == "Compare":
    render_compare(result, projects)
elif page == "Clean attention evidence":
    render_clean_attention(project_id)
else:
    render_graph(project_id)
