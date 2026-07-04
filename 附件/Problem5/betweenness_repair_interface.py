
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from build_osm_graph import load_osm_graph


def load_city_graph(csv_path: str | Path) -> nx.Graph:
    return load_osm_graph(Path(csv_path))


def extract_city_name(csv_path: str | Path) -> str:
    path = Path(csv_path)
    stem = path.stem
    suffix = "_Edgelist"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def radius_token_for_analysis(radius: float) -> str:
    rounded = round(radius)
    if abs(radius - rounded) > 1e-9:
        raise ValueError(
            "For compatibility with attack_utils.parse_attack_metadata, radius must be an integer value."
        )
    return str(int(rounded))


def ensure_xy_coords(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    coords: Dict[int, Tuple[float, float]] = {}
    missing_nodes: List[int] = []
    for node in graph.nodes():
        node_data = graph.nodes[node]
        if "x" not in node_data or "y" not in node_data:
            missing_nodes.append(node)
        else:
            coords[node] = (float(node_data["x"]), float(node_data["y"]))
    if missing_nodes:
        raise ValueError(
            f"All nodes must have x/y coordinates, but {len(missing_nodes)} nodes are missing coordinates."
        )
    return coords


def largest_component_nodes(graph: nx.Graph) -> set[int]:
    if graph.number_of_nodes() == 0:
        return set()
    return set(max(nx.connected_components(graph), key=len))


def largest_component_ratio(graph: nx.Graph, n_original: int) -> float:
    if n_original <= 0 or graph.number_of_nodes() == 0:
        return 0.0
    return len(largest_component_nodes(graph)) / n_original


class RadiusRemovalHelper:
    def __init__(self, graph: nx.Graph, radius: float) -> None:
        if radius < 0:
            raise ValueError("radius must be non-negative.")
        self.radius = float(radius)
        self.radius_sq = self.radius * self.radius
        self.cache: Dict[int, List[int]] = {}
        self.coords: Dict[int, Tuple[float, float]] = {}
        if self.radius > 0:
            self.coords = ensure_xy_coords(graph)

    def nodes_to_remove(self, center: int, working_graph: nx.Graph) -> List[int]:
        if self.radius <= 0:
            return [center] if center in working_graph else []
        if center not in self.cache:
            cx, cy = self.coords[center]
            affected: List[int] = []
            for node, (x, y) in self.coords.items():
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy <= self.radius_sq + 1e-12:
                    affected.append(node)
            affected.sort()
            self.cache[center] = affected
        return [node for node in self.cache[center] if node in working_graph]


@dataclass
class AttackRunResult:
    attack_sequence: List[int]
    attack_steps: int
    y_start: float
    y_end: float
    y_drop: float
    graph_before: nx.Graph
    graph_after: nx.Graph
    pre_window_lcc_nodes: set[int]


def _recompute_betweenness_ranking(
    working_graph: nx.Graph,
    normalized: bool,
    sample_k: int,
    random_seed: int,
) -> List[int]:
    current_n = working_graph.number_of_nodes()
    if current_n == 0:
        return []
    current_k = min(sample_k, current_n)
    bc = nx.betweenness_centrality(
        working_graph,
        k=current_k,
        normalized=normalized,
        seed=random_seed,
    )
    return [node for node, _ in sorted(bc.items(), key=lambda item: (-item[1], item[0]))]


def run_betweenness_attack_for_steps(
    input_graph: nx.Graph,
    attack_steps: int,
    n_original: int,
    radius: float = 0.0,
    normalized: bool = False,
    sample_k: int = 20,
    jump_step: int = 10,
    random_seed: int = 42,
) -> AttackRunResult:
    if attack_steps < 0:
        raise ValueError("attack_steps must be non-negative.")
    if sample_k <= 0:
        raise ValueError("sample_k must be positive.")
    if jump_step <= 0:
        raise ValueError("jump_step must be positive.")

    graph_before = input_graph.copy()
    working_graph = input_graph.copy()
    radius_helper = RadiusRemovalHelper(input_graph, radius)

    attack_sequence: List[int] = []
    y_start = largest_component_ratio(graph_before, n_original)
    pre_window_lcc_nodes = largest_component_nodes(graph_before)

    ranking: List[int] = []
    ranking_ptr = 0

    limit = min(attack_steps, working_graph.number_of_nodes())
    for step in range(limit):
        if working_graph.number_of_nodes() == 0:
            break
        if step % jump_step == 0 or ranking_ptr >= len(ranking):
            ranking = _recompute_betweenness_ranking(
                working_graph=working_graph,
                normalized=normalized,
                sample_k=sample_k,
                random_seed=random_seed,
            )
            ranking_ptr = 0

        while ranking_ptr < len(ranking) and ranking[ranking_ptr] not in working_graph:
            ranking_ptr += 1

        if ranking_ptr >= len(ranking):
            ranking = _recompute_betweenness_ranking(
                working_graph=working_graph,
                normalized=normalized,
                sample_k=sample_k,
                random_seed=random_seed,
            )
            ranking_ptr = 0
            while ranking_ptr < len(ranking) and ranking[ranking_ptr] not in working_graph:
                ranking_ptr += 1
            if ranking_ptr >= len(ranking):
                break

        target_node = ranking[ranking_ptr]
        ranking_ptr += 1
        attack_sequence.append(target_node)

        nodes_to_remove = radius_helper.nodes_to_remove(target_node, working_graph)
        if nodes_to_remove:
            working_graph.remove_nodes_from(nodes_to_remove)

    y_end = largest_component_ratio(working_graph, n_original)
    return AttackRunResult(
        attack_sequence=attack_sequence,
        attack_steps=len(attack_sequence),
        y_start=y_start,
        y_end=y_end,
        y_drop=y_start - y_end,
        graph_before=graph_before,
        graph_after=working_graph,
        pre_window_lcc_nodes=pre_window_lcc_nodes,
    )


def get_split_components(
    pre_window_lcc_nodes: set[int],
    after_graph: nx.Graph,
) -> List[set[int]]:
    surviving = pre_window_lcc_nodes & set(after_graph.nodes())
    if not surviving:
        return []
    subgraph = after_graph.subgraph(surviving)
    return [set(comp) for comp in nx.connected_components(subgraph)]


def find_component_pair_shortest_edges_kdtree(
    components: Sequence[set[int]],
    coords: Dict[int, Tuple[float, float]],
) -> List[Tuple[float, int, int, int, int]]:
    component_list = [sorted(comp) for comp in components if comp]
    if len(component_list) <= 1:
        return []

    candidates: List[Tuple[float, int, int, int, int]] = []
    for i in range(len(component_list)):
        nodes_i = component_list[i]
        pts_i = np.array([coords[node] for node in nodes_i], dtype=float)

        for j in range(i + 1, len(component_list)):
            nodes_j = component_list[j]
            pts_j = np.array([coords[node] for node in nodes_j], dtype=float)

            if len(nodes_i) <= len(nodes_j):
                query_nodes, query_pts = nodes_i, pts_i
                tree_nodes, tree_pts = nodes_j, pts_j
                swapped = False
            else:
                query_nodes, query_pts = nodes_j, pts_j
                tree_nodes, tree_pts = nodes_i, pts_i
                swapped = True

            tree = cKDTree(tree_pts)
            distances, indices = tree.query(query_pts, k=1)
            if np.isscalar(distances):
                distances = np.array([float(distances)])
                indices = np.array([int(indices)])

            best_idx = int(np.argmin(distances))
            best_distance = float(distances[best_idx])
            a = int(query_nodes[best_idx])
            b = int(tree_nodes[int(indices[best_idx])])

            if swapped:
                u, v = b, a
            else:
                u, v = a, b
            candidates.append((best_distance, u, v, i, j))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def choose_edges_to_add_disjoint_endpoints(
    current_graph: nx.Graph,
    components: Sequence[set[int]],
    coords: Dict[int, Tuple[float, float]],
    add_edge_count: int,
) -> List[Tuple[int, int, float]]:
    if add_edge_count <= 0:
        return []
    candidate_edges = find_component_pair_shortest_edges_kdtree(components=components, coords=coords)
    if not candidate_edges:
        return []

    selected: List[Tuple[int, int, float]] = []
    used_nodes: set[int] = set()
    selected_pairs: set[Tuple[int, int]] = set()

    for distance, u, v, _, _ in candidate_edges:
        pair = (min(u, v), max(u, v))
        if pair in selected_pairs:
            continue
        if current_graph.has_edge(u, v):
            continue
        if u in used_nodes or v in used_nodes:
            continue
        selected.append((u, v, distance))
        used_nodes.add(u)
        used_nodes.add(v)
        selected_pairs.add(pair)
        if len(selected) >= add_edge_count:
            break
    return selected


@dataclass
class WindowResult:
    window_index: int
    accepted_attack_sequence: List[int]
    accepted_attack_steps: int
    accepted_after_graph: nx.Graph
    accepted_added_edges: List[Tuple[int, int, float, int]]
    final_y_drop: float
    round_logs: List[Dict[str, Any]]


def run_single_window(
    window_initial_graph: nx.Graph,
    window_size: int,
    n_original: int,
    window_drop_threshold: float,
    add_edge_count_per_round: int,
    coords: Dict[int, Tuple[float, float]],
    window_index: int,
    radius: float = 0.0,
    normalized: bool = False,
    sample_k: int = 20,
    jump_step: int = 10,
    random_seed: int = 42,
    verbose: bool = False,
) -> WindowResult:
    current_start_graph = window_initial_graph.copy()
    accepted_added_edges: List[Tuple[int, int, float, int]] = []
    round_logs: List[Dict[str, Any]] = []
    repair_round = 0

    while True:
        repair_round += 1
        run = run_betweenness_attack_for_steps(
            input_graph=current_start_graph,
            attack_steps=window_size,
            n_original=n_original,
            radius=radius,
            normalized=normalized,
            sample_k=sample_k,
            jump_step=jump_step,
            random_seed=random_seed,
        )

        log_item: Dict[str, Any] = {
            "round_index": repair_round,
            "attack_steps": run.attack_steps,
            "y_start": run.y_start,
            "y_end": run.y_end,
            "y_drop": run.y_drop,
            "new_edges_added": 0,
            "new_edge_total_length": 0.0,
            "accepted": run.y_drop <= window_drop_threshold,
        }
        round_logs.append(log_item)

        if verbose:
            print(f"[window {window_index}] round {repair_round}: steps={run.attack_steps}, y_drop={run.y_drop:.6f}")

        if run.y_drop <= window_drop_threshold:
            return WindowResult(
                window_index=window_index,
                accepted_attack_sequence=run.attack_sequence,
                accepted_attack_steps=run.attack_steps,
                accepted_after_graph=run.graph_after,
                accepted_added_edges=accepted_added_edges,
                final_y_drop=run.y_drop,
                round_logs=round_logs,
            )

        components = get_split_components(run.pre_window_lcc_nodes, run.graph_after)
        if len(components) <= 1:
            return WindowResult(
                window_index=window_index,
                accepted_attack_sequence=run.attack_sequence,
                accepted_attack_steps=run.attack_steps,
                accepted_after_graph=run.graph_after,
                accepted_added_edges=accepted_added_edges,
                final_y_drop=run.y_drop,
                round_logs=round_logs,
            )

        proposed_edges = choose_edges_to_add_disjoint_endpoints(
            current_graph=run.graph_after,
            components=components,
            coords=coords,
            add_edge_count=add_edge_count_per_round,
        )
        if not proposed_edges:
            return WindowResult(
                window_index=window_index,
                accepted_attack_sequence=run.attack_sequence,
                accepted_attack_steps=run.attack_steps,
                accepted_after_graph=run.graph_after,
                accepted_added_edges=accepted_added_edges,
                final_y_drop=run.y_drop,
                round_logs=round_logs,
            )

        log_item["new_edges_added"] = len(proposed_edges)
        log_item["new_edge_total_length"] = float(sum(length for _, _, length in proposed_edges))
        for u, v, length in proposed_edges:
            current_start_graph.add_edge(
                u, v,
                length=float(length),
                added_by="window_repair",
                window_index=window_index,
                repair_round=repair_round,
            )
            accepted_added_edges.append((u, v, float(length), repair_round))


def save_attack_sequence_csv(output_path: Path, attack_sequence: Sequence[int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "node"])
        for idx, node in enumerate(attack_sequence, start=1):
            writer.writerow([idx, node])


def save_added_edges_csv(
    output_path: Path,
    added_edges: Sequence[Tuple[int, int, float, int, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["edge_index", "u", "v", "length", "window_index", "repair_round"])
        for idx, (u, v, length, window_index, repair_round) in enumerate(added_edges, start=1):
            writer.writerow([idx, u, v, length, window_index, repair_round])


def save_summary_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_betweenness_repair_interface(
    graph_csv_path: str | Path,
    sample_k: int,
    jump_step: int,
    window_drop_slope: float,
    end_y_threshold: float,
    radius: float,
    add_edge_count_per_round: int,
    output_root: str | Path,
    *,
    normalized: bool = False,
    random_seed: int = 42,
    verbose: bool = False,
) -> Dict[str, str]:
    if sample_k <= 0:
        raise ValueError("sample_k must be positive.")
    if jump_step <= 0:
        raise ValueError("jump_step must be positive.")
    if window_drop_slope < 0:
        raise ValueError("window_drop_slope must be non-negative.")
    if not (0 <= end_y_threshold <= 1):
        raise ValueError("end_y_threshold must be between 0 and 1.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if add_edge_count_per_round <= 0:
        raise ValueError("add_edge_count_per_round must be positive.")

    graph_csv_path = Path(graph_csv_path)
    output_root = Path(output_root)

    city_name = extract_city_name(graph_csv_path)
    radius_token = radius_token_for_analysis(radius)
    strategy_token = "Betweenness"
    folder_name = f"{city_name}{strategy_token}{radius_token}"
    output_dir = output_root / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = load_city_graph(graph_csv_path)
    n_original = graph.number_of_nodes()
    window_size = jump_step
    window_drop_threshold = window_drop_slope * (window_size / n_original)
    coords = ensure_xy_coords(graph)

    full_repaired_graph = graph.copy()
    current_window_initial_graph = full_repaired_graph.copy()

    final_attack_sequence: List[int] = []
    all_added_edges: List[Tuple[int, int, float, int, int]] = []
    all_window_logs: List[Dict[str, Any]] = []

    window_index = 0
    while current_window_initial_graph.number_of_nodes() > 0:
        window_index += 1
        window_result = run_single_window(
            window_initial_graph=current_window_initial_graph,
            window_size=window_size,
            n_original=n_original,
            window_drop_threshold=window_drop_threshold,
            add_edge_count_per_round=add_edge_count_per_round,
            coords=coords,
            window_index=window_index,
            radius=radius,
            normalized=normalized,
            sample_k=sample_k,
            jump_step=jump_step,
            random_seed=random_seed,
            verbose=verbose,
        )

        for u, v, length, repair_round in window_result.accepted_added_edges:
            full_repaired_graph.add_edge(
                u, v,
                length=float(length),
                added_by="window_repair",
                window_index=window_index,
                repair_round=repair_round,
            )
            all_added_edges.append((u, v, float(length), window_index, repair_round))

        final_attack_sequence.extend(window_result.accepted_attack_sequence)
        current_window_initial_graph = window_result.accepted_after_graph

        current_y = largest_component_ratio(current_window_initial_graph, n_original)
        all_window_logs.append(
            {
                "window_index": window_result.window_index,
                "accepted_attack_steps": window_result.accepted_attack_steps,
                "accepted_y_end": current_y,
                "final_y_drop": window_result.final_y_drop,
                "round_logs": window_result.round_logs,
            }
        )

        if current_y < end_y_threshold:
            break
        if window_result.accepted_attack_steps == 0:
            break

    attack_csv_name = f"{city_name}{strategy_token}{radius_token}.csv"
    attack_csv_path = output_dir / attack_csv_name
    edges_csv_path = output_dir / f"{city_name}{strategy_token}{radius_token}_added_edges.csv"
    summary_json_path = output_dir / f"{city_name}{strategy_token}{radius_token}_summary.json"

    save_attack_sequence_csv(attack_csv_path, final_attack_sequence)
    save_added_edges_csv(edges_csv_path, all_added_edges)

    total_added_edge_count = len(all_added_edges)
    total_added_edge_length = float(sum(length for _, _, length, _, _ in all_added_edges))

    summary_payload: Dict[str, Any] = {
        "city_name": city_name,
        "graph_csv_path": str(graph_csv_path),
        "output_dir": str(output_dir),
        "attack_sequence_csv": str(attack_csv_path),
        "added_edges_csv": str(edges_csv_path),
        "sample_k": sample_k,
        "jump_step": jump_step,
        "window_size": window_size,
        "window_drop_slope": window_drop_slope,
        "window_drop_threshold_effective": window_drop_threshold,
        "end_y_threshold": end_y_threshold,
        "radius": radius,
        "normalized": normalized,
        "random_seed": random_seed,
        "attack_sequence_length": len(final_attack_sequence),
        "total_added_edge_count": total_added_edge_count,
        "total_added_edge_length": total_added_edge_length,
        "window_count": len(all_window_logs),
        "effective_strategy_token_for_analysis": strategy_token,
        "analysis_filename_example": attack_csv_name,
    }
    save_summary_json(summary_json_path, summary_payload)

    return {
        "attack_sequence_csv": str(attack_csv_path),
        "added_edges_csv": str(edges_csv_path),
        "summary_json": str(summary_json_path),
        "output_dir": str(output_dir),
    }


def main() -> None:
    outputs = run_betweenness_repair_interface(
        graph_csv_path="Chengdu_Edgelist.csv",
        sample_k=20,
        jump_step=10,
        window_drop_slope=10.0,
        end_y_threshold=0.01,
        radius=0.0,
        add_edge_count_per_round=5,
        output_root="results_interface",
        normalized=False,
        random_seed=42,
        verbose=True,
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
