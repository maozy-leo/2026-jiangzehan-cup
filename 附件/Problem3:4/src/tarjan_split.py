"""Tarjan articulation-based split gain computation."""
from __future__ import annotations

import sys
from typing import Dict

import networkx as nx


def split_gain(graph: nx.Graph) -> Dict[int, int]:
    """Return max(0, cc(G-v) - cc(G)) for every node using Tarjan DFS."""

    target_limit = max(10000, graph.number_of_nodes() * 2)
    if sys.getrecursionlimit() < target_limit:
        sys.setrecursionlimit(target_limit)
    time = 0
    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    gain: Dict[int, int] = {node: 0 for node in graph.nodes}

    def dfs(u: int) -> None:
        nonlocal time
        time += 1
        disc[u] = low[u] = time
        child_cnt = 0

        for v in graph.neighbors(u):
            if v not in disc:
                parent[v] = u
                child_cnt += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None:
                    gain[u] = max(0, child_cnt - 1)
                else:
                    if low[v] >= disc[u]:
                        gain[u] += 1
            elif parent.get(u) != v:
                low[u] = min(low[u], disc[v])

    for node in graph.nodes:
        if node not in disc:
            dfs(node)

    return gain
