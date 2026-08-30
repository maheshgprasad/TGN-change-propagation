"""Reconstruct a co-change graph from author ChangeSets (read-only).

The trained TGN never writes NetworkX graphs, adjacency matrices, or edge lists.
This module rebuilds directed co-changeability with the same formula as
TGN_model.py so the dashboard can show the input graph the model uses.

Original formula (TGN_model.py):
    start = max(first_commit(A), first_commit(B))
    score(A→B) = |commits(A) ∩ commits(B)| / |commits(A)|   within [start, end]
The function named `union` only counts file A's commits, so the score is directed.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict

import networkx as nx
from pyvis.network import Network

from data_loader import filter_changesets_like_authors, load_raw_changesets


def file_label(file_id: str) -> str:
    return f"File {file_id}"


def count_in_range(values: list[int], start: int, end: int) -> int:
    return bisect_right(values, end) - bisect_left(values, start)


def cochangeability(
    file_a: str,
    file_b: str,
    end_time: int,
    changes_of_file: dict[str, list[int]],
    changes_set: dict[str, set[int]],
) -> float:
    a_values = changes_of_file[file_a]
    b_values = changes_of_file[file_b]
    start = max(a_values[0], b_values[0])
    denominator = count_in_range(a_values, start, end_time)
    if denominator == 0:
        return 0.0
    left = bisect_left(b_values, start)
    right = bisect_right(b_values, end_time)
    common = sum(index in changes_set[file_a] for index in b_values[left:right])
    return common / denominator


def history_up_to(
    project: str,
    shuffle: int,
    commit_index: int | None = None,
) -> tuple[list[list[str]], dict[str, list[int]], dict[str, set[int]]] | None:
    raw = load_raw_changesets(project, shuffle)
    if raw is None:
        return None
    changes = filter_changesets_like_authors(raw)
    if not changes:
        return None
    if commit_index is None:
        commit_index = len(changes) - 1
    commit_index = max(0, min(commit_index, len(changes) - 1))

    changes_of_file: dict[str, list[int]] = defaultdict(list)
    changes_set: dict[str, set[int]] = defaultdict(set)
    for index, commit in enumerate(changes[: commit_index + 1]):
        for filename in commit[1:]:
            changes_of_file[filename].append(index)
            changes_set[filename].add(index)
    return changes, changes_of_file, changes_set


def ranked_outgoing(
    source_file: str,
    changes_of_file: dict[str, list[int]],
    changes_set: dict[str, set[int]],
    end_time: int,
    min_weight: float,
    limit: int,
) -> list[tuple[str, float]]:
    neighbors: list[tuple[str, float]] = []
    for target in changes_of_file:
        if target == source_file:
            continue
        weight = cochangeability(
            source_file, target, end_time, changes_of_file, changes_set
        )
        if weight >= min_weight:
            neighbors.append((target, weight))
    neighbors.sort(key=lambda item: item[1], reverse=True)
    return neighbors[: max(0, limit)]


def build_subgraph(
    changes_of_file: dict[str, list[int]],
    changes_set: dict[str, set[int]],
    end_time: int,
    min_weight: float,
    max_nodes: int,
    source_file: str | None = None,
) -> nx.DiGraph:
    ranked = sorted(changes_of_file, key=lambda name: len(changes_of_file[name]), reverse=True)
    graph = nx.DiGraph()

    if source_file and source_file in changes_of_file:
        # Ego graph only: source → companions. Pairwise edges among
        # companions turn this into an unreadable hairball.
        neighbors = ranked_outgoing(
            source_file,
            changes_of_file,
            changes_set,
            end_time,
            min_weight,
            max(0, max_nodes - 1),
        )
        graph.add_node(source_file)
        for target, weight in neighbors:
            graph.add_edge(source_file, target, weight=weight)
        return graph

    keep = ranked[: max(1, max_nodes)]
    graph.add_nodes_from(keep)
    for src in keep:
        for dst in keep:
            if src == dst:
                continue
            weight = cochangeability(src, dst, end_time, changes_of_file, changes_set)
            if weight >= min_weight:
                graph.add_edge(src, dst, weight=weight)
    return graph


def neighbor_rows(
    graph: nx.DiGraph,
    source_file: str,
    changes_of_file: dict[str, list[int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    outgoing = sorted(
        graph.out_edges(source_file, data=True),
        key=lambda item: float(item[2].get("weight", 0.0)),
        reverse=True,
    )
    for rank, (_, target, data) in enumerate(outgoing, start=1):
        weight = float(data.get("weight", 0.0))
        rows.append(
            {
                "Rank": rank,
                "File": file_label(target),
                "File ID": target,
                "Co-change score": weight,
                "When the source changed, this file was in the same commit": f"{weight:.0%}",
                "Times this file changed in this history slice": len(
                    changes_of_file.get(target, [])
                ),
            }
        )
    return rows


def graph_html(
    graph: nx.DiGraph,
    source_file: str | None = None,
    change_counts: dict[str, int] | None = None,
    height: str = "560px",
) -> str:
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1b2631",
        cdn_resources="in_line",
    )
    counts = change_counts or {}
    neighbors = set(graph.successors(source_file)) if source_file in graph else set()
    count_values = [counts.get(node, 1) for node in graph.nodes()] or [1]
    max_count = max(count_values)

    for node in graph.nodes():
        n_changes = counts.get(node, 0)
        size = 18 + 22 * (n_changes / max_count if max_count else 0)
        if node == source_file:
            color = "#c0392b"
            role = "Selected source file — the file we treat as just changed"
            level = 0
        elif node in neighbors:
            color = "#d35400"
            role = "Historically co-changed with the source"
            level = 1
        else:
            color = "#1f4e79"
            role = "File in this history slice"
            level = 1
        title = (
            f"{file_label(node)}<br>"
            f"{role}<br>"
            f"Appeared in {n_changes} commit(s) up to this slice.<br>"
            "IDs come from the change-set CSV; the authors do not store file paths here."
        )
        net.add_node(
            node,
            label=file_label(node),
            title=title,
            color=color,
            size=size,
            level=level,
        )

    for src, dst, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.0))
        title = (
            f"When {file_label(src)} changed, {file_label(dst)} was in the same "
            f"commit {weight:.0%} of the time (score {weight:.3f})."
        )
        net.add_edge(
            src,
            dst,
            value=max(weight, 0.08),
            title=title,
            label=f"{weight:.2f}" if weight >= 0.25 else "",
            arrows="to",
            color="#7f8c8d",
        )

    if source_file and source_file in graph:
        net.set_options(
            """
            {
              "nodes": {"font": {"size": 14, "face": "arial"}, "borderWidth": 2},
              "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
                "font": {"size": 12, "color": "#1f4e79", "strokeWidth": 0},
                "smooth": false
              },
              "layout": {
                "hierarchical": {
                  "enabled": true,
                  "direction": "LR",
                  "levelSeparation": 260,
                  "nodeSpacing": 70,
                  "sortMethod": "hubsize"
                }
              },
              "physics": {"enabled": false}
            }
            """
        )
    else:
        net.set_options(
            """
            {
              "nodes": {"font": {"size": 13, "face": "arial"}},
              "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.55}},
                "smooth": {"type": "continuous"}
              },
              "physics": {
                "barnesHut": {"gravitationalConstant": -12000, "springLength": 160},
                "stabilization": {"iterations": 80}
              }
            }
            """
        )

    return net.generate_html()
