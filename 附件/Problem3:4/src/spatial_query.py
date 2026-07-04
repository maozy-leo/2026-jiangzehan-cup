"""Spatial helpers for radius-based failures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Coordinates = Dict[int, Tuple[float, float]]


def brute_force_nodes_within_radius(
    node: int,
    radius: float,
    coords: Coordinates,
    candidates: Iterable[int],
) -> List[int]:
    if radius <= 0:
        return [node]
    center = coords.get(node)
    if center is None:
        return [node]
    cx, cy = center
    radius_sq = radius * radius
    hits: List[int] = []
    for other in candidates:
        other_coord = coords.get(other)
        if other_coord is None:
            if other == node:
                hits.append(other)
            continue
        dx = other_coord[0] - cx
        dy = other_coord[1] - cy
        if dx * dx + dy * dy <= radius_sq:
            hits.append(other)
    if node not in hits:
        hits.append(node)
    return hits


class SpatialIndex:
    """Radius helper that optionally leverages a KD-tree."""

    def __init__(self, coords: Coordinates, method: str = "bruteforce", rebuild: bool = False) -> None:
        self.coords = coords
        self.method = (method or "bruteforce").lower()
        if self.method not in {"bruteforce", "kdtree"}:
            raise ValueError(f"Unsupported spatial_index: {method}")
        self.rebuild = rebuild
        self._active_nodes: List[int] = []
        self._kd_tree: Optional[_KDTreeIndex] = None
        if self.method == "kdtree":
            self._kd_tree = _KDTreeIndex(coords)

    def update_active_nodes(self, active_nodes: Iterable[int]) -> None:
        self._active_nodes = list(active_nodes)
        if self.method == "kdtree" and self.rebuild:
            subset = {node: self.coords[node] for node in self._active_nodes if node in self.coords}
            self._kd_tree = _KDTreeIndex(subset) if subset else None

    def nodes_within_radius(self, node: int, radius: float) -> List[int]:
        if radius <= 0:
            return [node]
        if node not in self.coords:
            return [node]
        if self.method == "kdtree" and self._kd_tree is not None:
            hits = self._kd_tree.query_radius(self.coords[node], radius)
            if self._active_nodes:
                active_set = set(self._active_nodes)
                hits = [candidate for candidate in hits if candidate in active_set]
            if node not in hits:
                hits.append(node)
            return hits or [node]
        candidates = self._active_nodes if self._active_nodes else list(self.coords.keys())
        return brute_force_nodes_within_radius(node, radius, self.coords, candidates)


@dataclass
class _KDTreeNode:
    point: Tuple[float, float, int]
    axis: int
    left: Optional["_KDTreeNode"] = None
    right: Optional["_KDTreeNode"] = None


class _KDTreeIndex:
    def __init__(self, coords: Coordinates) -> None:
        points = [(float(x), float(y), node) for node, (x, y) in coords.items()]
        self.root = self._build(points, depth=0)

    def _build(self, points: Sequence[Tuple[float, float, int]], depth: int) -> Optional[_KDTreeNode]:
        if not points:
            return None
        axis = depth % 2
        sorted_points = sorted(points, key=lambda item: item[axis])
        median = len(sorted_points) // 2
        node = _KDTreeNode(sorted_points[median], axis)
        node.left = self._build(sorted_points[:median], depth + 1)
        node.right = self._build(sorted_points[median + 1 :], depth + 1)
        return node

    def query_radius(self, center: Tuple[float, float], radius: float) -> List[int]:
        results: List[int] = []
        if self.root is None:
            return results
        radius_sq = radius * radius
        self._search(self.root, center, radius, radius_sq, results)
        return results

    def _search(
        self,
        node: Optional[_KDTreeNode],
        center: Tuple[float, float],
        radius: float,
        radius_sq: float,
        results: List[int],
    ) -> None:
        if node is None:
            return
        px, py, idx = node.point
        dx = center[0] - px
        dy = center[1] - py
        if dx * dx + dy * dy <= radius_sq:
            results.append(idx)
        axis = node.axis
        diff = center[axis] - (px if axis == 0 else py)
        if diff <= radius:
            self._search(node.left, center, radius, radius_sq, results)
        if diff >= -radius:
            self._search(node.right, center, radius, radius_sq, results)
