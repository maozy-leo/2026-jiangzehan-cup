from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - SciPy may be missing on some systems
    cKDTree = None  # type: ignore

from build_osm_graph import load_osm_graph

CITY_NAMES = (
    "Chengdu",
    "Dalian",
    "Dongguan",
    "Harbin",
    "Qingdao",
    "Quanzhou",
    "Shenyang",
    "Zhengzhou",
)


@dataclass(frozen=True)
class AttackMetadata:
    csv_path: Path
    city: str
    strategy: str
    radius: float
    graph_csv: Path


def parse_attack_metadata(csv_path: str | Path, graph_dir: str | Path = "data") -> AttackMetadata:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Attack CSV not found: {path}")

    stem = path.stem
    m = re.search(r"(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse attack radius from file name: {path.name}")
    radius = float(m.group(1))
    prefix = stem[: m.start()]

    city = None
    for candidate in sorted(CITY_NAMES, key=len, reverse=True):
        if prefix.startswith(candidate):
            city = candidate
            break
    if city is None:
        raise ValueError(f"Cannot determine city from file name: {path.name}")

    strategy = prefix[len(city) :]
    graph_csv = Path(graph_dir) / f"{city}_Edgelist.csv"
    if not graph_csv.exists():
        raise FileNotFoundError(f"Graph CSV does not exist: {graph_csv}")

    return AttackMetadata(
        csv_path=path,
        city=city,
        strategy=strategy or "",
        radius=radius,
        graph_csv=graph_csv,
    )


def load_attack_sequence(csv_path: str | Path) -> List[int]:
    import csv

    attack_sequence: List[int] = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "node" not in reader.fieldnames:
            raise ValueError("Attack CSV must include a 'node' column.")
        for row in reader:
            try:
                attack_sequence.append(int(row["node"]))
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid node value in {csv_path}: {row['node']}") from exc

    if not attack_sequence:
        raise ValueError(f"Attack CSV {csv_path} is empty.")

    return attack_sequence


@dataclass
class SpatialIndex:
    tree: cKDTree | None
    node_ids: np.ndarray
    node_to_index: Dict[int, int]
    coords: np.ndarray

    def nodes_within_radius(self, node_id: int, radius: float) -> List[int]:
        if self.tree is None or radius <= 0:
            return []
        idx = self.node_to_index.get(node_id)
        if idx is None:
            return []
        point = self.coords[idx]
        if np.isnan(point).any():
            return []
        candidate_idxs = self.tree.query_ball_point(point, r=radius)
        return [int(self.node_ids[i]) for i in candidate_idxs]


def build_spatial_index(graph: nx.Graph) -> SpatialIndex:
    nodes_with_coords: List[int] = []
    coords: List[Tuple[float, float]] = []
    for node, data in graph.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        nodes_with_coords.append(int(node))
        coords.append((float(x), float(y)))

    if not nodes_with_coords or cKDTree is None:
        return SpatialIndex(tree=None, node_ids=np.array([]), node_to_index={}, coords=np.zeros((0, 2)))

    coord_array = np.asarray(coords, dtype=float)
    tree = cKDTree(coord_array)
    node_ids_array = np.asarray(nodes_with_coords, dtype=int)
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes_with_coords)}
    return SpatialIndex(tree=tree, node_ids=node_ids_array, node_to_index=node_to_index, coords=coord_array)


@lru_cache(maxsize=16)
def load_city_graph(graph_csv: str | Path) -> nx.Graph:
    return load_osm_graph(Path(graph_csv))


@dataclass
class AttackStepMetrics:
    step: int
    removed_nodes: int
    removed_ratio: float
    largest_component_size: int
    largest_component_ratio: float


def simulate_attack(
    graph: nx.Graph,
    attack_sequence: Sequence[int],
    radius: float,
    *,
    record_steps: Iterable[int] | None = None,
) -> Tuple[List[AttackStepMetrics], Dict[int, nx.Graph], Dict[int, Set[int]]]:
    if radius < 0:
        raise ValueError("Attack radius must be non-negative.")

    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        raise ValueError("Graph is empty.")

    current_graph = graph.copy()
    removed_count = 0
    spatial_index = build_spatial_index(graph)

    record_steps_set: Set[int] = set(record_steps or [])
    graphs_by_step: Dict[int, nx.Graph] = {}
    lcc_nodes_by_step: Dict[int, Set[int]] = {}
    metrics: List[AttackStepMetrics] = []

    initial_lcc_nodes = max(nx.connected_components(current_graph), key=len)
    initial_lcc_size = len(initial_lcc_nodes)
    metrics.append(
        AttackStepMetrics(
            step=0,
            removed_nodes=0,
            removed_ratio=0.0,
            largest_component_size=initial_lcc_size,
            largest_component_ratio=initial_lcc_size / total_nodes,
        )
    )
    if 0 in record_steps_set:
        graphs_by_step[0] = current_graph.copy()
        lcc_nodes_by_step[0] = set(initial_lcc_nodes)

    for step_idx, node_id in enumerate(attack_sequence, start=1):
        nodes_to_remove: Set[int] = set()
        if node_id in current_graph:
            nodes_to_remove.add(int(node_id))
        nodes_to_remove.update(spatial_index.nodes_within_radius(int(node_id), radius))
        nodes_to_remove = {n for n in nodes_to_remove if n in current_graph}

        if nodes_to_remove:
            current_graph.remove_nodes_from(nodes_to_remove)
            removed_count += len(nodes_to_remove)

        if current_graph.number_of_nodes() == 0:
            largest_nodes: Set[int] = set()
            largest_size = 0
        else:
            largest_nodes = set(max(nx.connected_components(current_graph), key=len))
            largest_size = len(largest_nodes)

        metrics.append(
            AttackStepMetrics(
                step=step_idx,
                removed_nodes=removed_count,
                removed_ratio=removed_count / total_nodes,
                largest_component_size=largest_size,
                largest_component_ratio=largest_size / total_nodes,
            )
        )

        if step_idx in record_steps_set:
            graphs_by_step[step_idx] = current_graph.copy()
            lcc_nodes_by_step[step_idx] = largest_nodes

        if largest_size == 0:
            # No nodes left; remaining steps won't change anything
            continue

    return metrics, graphs_by_step, lcc_nodes_by_step
