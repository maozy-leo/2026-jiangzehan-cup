"""Compute graph statistics for each city OSM edge list.

This script reuses ``load_osm_graph`` so we only have a single source of
truth for cleaning edges. For every ``*_Edgelist.csv`` file it finds (or the
explicit set passed via CLI) it will

1. build the undirected NetworkX graph;
2. compute clustering metrics, path-length metrics (always using exact
   shortest-path computations), connected component data, and the degree
   distribution; and
3. write one CSV per city that contains all scalar metrics followed by
   degree-count rows (``city_tables/<city>_stats.csv``).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx

from build_osm_graph import load_osm_graph


@dataclass(slots=True)
class CityGraphStats:
    city: str
    metrics: dict[str, float | int]
    degree_distribution: Sequence[tuple[int, int]]


@dataclass(slots=True)
class ComponentPathStats:
    num_nodes: int
    num_edges: int
    diameter: float
    avg_path_length: float
    pair_weight: float


SUMMARY_FIELDS = [
    "num_nodes",
    "num_edges",
    "avg_local_clustering",
    "global_transitivity",
    "num_connected_components",
    "network_diameter",
    "average_path_length",
    "largest_component_nodes",
    "largest_component_edges",
    "largest_component_diameter",
    "largest_component_avg_path_length",
    "largest_component_size_ratio",
    "min_degree",
    "max_degree",
]


def _city_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_Edgelist"):
        return stem[: -len("_Edgelist")]
    return stem


def _component_path_stats(subgraph: nx.Graph) -> ComponentPathStats:
    """Compute diameter and mean path length for a connected component."""

    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()

    if num_nodes == 0:
        return ComponentPathStats(
            num_nodes=0,
            num_edges=num_edges,
            diameter=math.nan,
            avg_path_length=math.nan,
            pair_weight=0.0,
        )

    if num_nodes == 1:
        return ComponentPathStats(
            num_nodes=1,
            num_edges=num_edges,
            diameter=0.0,
            avg_path_length=math.nan,
            pair_weight=0.0,
        )

    diameter = float(nx.diameter(subgraph, usebounds=True))
    pair_weight = float(num_nodes * (num_nodes - 1))

    avg_length = float(nx.average_shortest_path_length(subgraph))

    return ComponentPathStats(
        num_nodes=num_nodes,
        num_edges=num_edges,
        diameter=diameter,
        avg_path_length=avg_length,
        pair_weight=pair_weight,
    )


def _degree_distribution(graph: nx.Graph) -> list[tuple[int, int]]:
    counts = Counter(dict(graph.degree()).values())
    return sorted(counts.items())


def compute_stats(graph: nx.Graph) -> tuple[dict[str, float | int], list[tuple[int, int]]]:
    metrics: dict[str, float | int] = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "avg_local_clustering": nx.average_clustering(graph),
        "global_transitivity": nx.transitivity(graph),
    }

    component_count = 0
    network_diameter = math.nan
    weighted_path_sum = 0.0
    total_ordered_pairs = 0.0
    largest_component: ComponentPathStats | None = None

    for component_nodes in nx.connected_components(graph):
        component_count += 1
        subgraph = graph.subgraph(component_nodes)
        comp_stats = _component_path_stats(subgraph)

        if math.isnan(network_diameter):
            network_diameter = comp_stats.diameter
        else:
            network_diameter = max(network_diameter, comp_stats.diameter)

        if comp_stats.num_nodes > 1 and not math.isnan(comp_stats.avg_path_length):
            weighted_path_sum += comp_stats.avg_path_length * comp_stats.pair_weight
            total_ordered_pairs += comp_stats.pair_weight

        if largest_component is None or comp_stats.num_nodes > largest_component.num_nodes:
            largest_component = comp_stats

    metrics["num_connected_components"] = component_count
    metrics["network_diameter"] = network_diameter
    metrics["average_path_length"] = (
        (weighted_path_sum / total_ordered_pairs) if total_ordered_pairs else math.nan
    )

    if largest_component:
        metrics["largest_component_nodes"] = largest_component.num_nodes
        metrics["largest_component_edges"] = largest_component.num_edges
        metrics["largest_component_diameter"] = largest_component.diameter
        metrics["largest_component_avg_path_length"] = largest_component.avg_path_length
    else:
        metrics["largest_component_nodes"] = 0
        metrics["largest_component_edges"] = 0
        metrics["largest_component_diameter"] = math.nan
        metrics["largest_component_avg_path_length"] = math.nan

    metrics["largest_component_size_ratio"] = (
        metrics["largest_component_nodes"] / metrics["num_nodes"]
        if metrics["num_nodes"]
        else math.nan
    )

    degree_distribution = _degree_distribution(graph)
    metrics["min_degree"] = min((deg for deg, _ in degree_distribution), default=math.nan)
    metrics["max_degree"] = max((deg for deg, _ in degree_distribution), default=math.nan)

    return metrics, degree_distribution


def _write_city_tables(stats: Iterable[CityGraphStats], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stat in stats:
        file_path = output_dir / f"{stat.city}_stats.csv"
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for key in SUMMARY_FIELDS:
                writer.writerow([key, stat.metrics.get(key, math.nan)])
            for key, value in stat.metrics.items():
                if key in SUMMARY_FIELDS:
                    continue
                writer.writerow([key, value])
            for degree, count in stat.degree_distribution:
                writer.writerow([f"degree_count_{degree}", count])


def discover_csv_paths(args_paths: Sequence[Path]) -> list[Path]:
    if args_paths:
        return [path if path.is_absolute() else Path.cwd() / path for path in args_paths]
    return sorted(Path.cwd().glob("*_Edgelist.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Optional list of specific edge-list CSV files to process.",
    )
    parser.add_argument(
        "--city-output-dir",
        type=Path,
        default=Path("city_tables"),
        help="Directory where per-city CSV tables will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = discover_csv_paths(args.csv_paths)
    if not csv_paths:
        raise SystemExit("No *_Edgelist.csv files found to process.")

    all_stats: list[CityGraphStats] = []
    for csv_path in csv_paths:
        city = _city_name(csv_path)
        graph = load_osm_graph(csv_path)
        metrics, degree_distribution = compute_stats(graph)
        print(
            f"[{city}] nodes={metrics['num_nodes']:,} edges={metrics['num_edges']:,} "
            f"components={metrics['num_connected_components']}",
            flush=True,
        )
        all_stats.append(CityGraphStats(city=city, metrics=metrics, degree_distribution=degree_distribution))

    _write_city_tables(all_stats, args.city_output_dir)


if __name__ == "__main__":
    main()
