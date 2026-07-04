"""Basic graph operations used by attack strategies."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set

import networkx as nx


def clone_graph(graph: nx.Graph) -> nx.Graph:
    """Return a fast shallow copy of the graph."""

    return graph.copy(as_view=False)


def remove_nodes(graph: nx.Graph, nodes: Iterable[int]) -> None:
    """Remove nodes if they exist."""

    graph.remove_nodes_from(list(nodes))


def get_connected_components(graph: nx.Graph) -> List[Set[int]]:
    """Return connected components sorted by size descending."""

    comps = list(nx.connected_components(graph))
    comps.sort(key=len, reverse=True)
    return comps


def get_lcc_nodes(graph: nx.Graph) -> Set[int]:
    """Return nodes of largest connected component (empty set if graph empty)."""

    comps = get_connected_components(graph)
    return comps[0] if comps else set()
