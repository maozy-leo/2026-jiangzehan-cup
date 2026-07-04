"""Proxy metrics and evaluation helpers."""
from __future__ import annotations

from typing import Tuple

import networkx as nx


def component_sizes(graph: nx.Graph) -> Tuple[int, ...]:
    if graph.number_of_nodes() == 0:
        return tuple()
    sizes = tuple(len(c) for c in nx.connected_components(graph))
    return sizes


def largest_component_ratio(graph: nx.Graph, total_nodes: int) -> float:
    if total_nodes == 0:
        return 0.0
    if graph.number_of_nodes() == 0:
        return 0.0
    largest = max(len(c) for c in nx.connected_components(graph))
    return largest / total_nodes


def fragmentation_score(graph: nx.Graph, total_nodes: int) -> float:
    if total_nodes == 0:
        return 0.0
    sizes = component_sizes(graph)
    if not sizes:
        return 0.0
    return sum(size * size for size in sizes) / (total_nodes * total_nodes)


def proxy_value(graph: nx.Graph, objective: str, total_nodes: int, alpha: float = 0.7) -> float:
    objective = objective.upper()
    if objective == "R1":
        return largest_component_ratio(graph, total_nodes)
    if objective == "R2":
        lcc = largest_component_ratio(graph, total_nodes)
        frag = fragmentation_score(graph, total_nodes)
        return alpha * lcc + (1.0 - alpha) * frag
    raise ValueError(f"Unsupported objective: {objective}")


def robustness_increment(p_t: float, removed_this_step: int, total_nodes: int) -> float:
    if total_nodes == 0:
        return 0.0
    return p_t * (removed_this_step / total_nodes)
