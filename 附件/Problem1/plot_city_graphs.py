"""Generate geographic and topological visualizations for every *_Edgelist.csv.

The script loads the cleaned undirected graph via ``build_osm_graph.load_osm_graph``
and creates two static PNGs per city:

1. Geographic overlay: projected coordinates drawn on top of a basemap
   (Contextily tiles when available, otherwise Natural Earth coastlines).
2. Topology layout: standard force-directed (spring) layout colored by
   connected component and scaled by node degree.

Example:

    /opt/homebrew/Caskroom/miniconda/base/envs/myenv/bin/python \
        plot_city_graphs.py --output-dir figures

"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_hex
from pyproj import CRS
from shapely.geometry import LineString, Point

try:
    import contextily as cx

    HAS_CONTEXTILY = True
except ModuleNotFoundError:  # pragma: no cover - informative fallback
    cx = None
    HAS_CONTEXTILY = False

from build_osm_graph import load_osm_graph


CITY_ZONE_LOOKUP: dict[str, int] = {
    "Chengdu": 48,
    "Dongguan": 49,
    "Dalian": 51,
    "Harbin": 52,
    "Qingdao": 51,
    "Quanzhou": 50,
    "Shenyang": 51,
    "Zhengzhou": 49,
}

DEFAULT_TILE_PROVIDER = "CartoDB.Voyager"
BASEMAP_FAILED = False
TOPOLOGY_LAYOUT_SEED = 42
TOPOLOGY_MAX_NODES = 5000
WORLD_BACKGROUND: gpd.GeoDataFrame | None = None
WORLD_BACKGROUND_ERROR = False


def _city_name(csv_path: Path) -> str:
    stem = csv_path.stem
    if stem.endswith("_Edgelist"):
        return stem[: -len("_Edgelist")]
    return stem


def _city_crs(city: str) -> CRS:
    zone = CITY_ZONE_LOOKUP.get(city)
    if zone is None:
        raise ValueError(
            f"UTM zone for city '{city}' is unknown. Update CITY_ZONE_LOOKUP first."
        )
    return CRS.from_epsg(32600 + zone)


def discover_csv_paths(args_paths: Iterable[Path]) -> list[Path]:
    if args_paths:
        return [path if path.is_absolute() else Path.cwd() / path for path in args_paths]
    return sorted(Path.cwd().glob("*_Edgelist.csv"))


def _component_maps(graph: nx.Graph) -> tuple[dict[int, int], dict[int, int]]:
    comp_map: dict[int, int] = {}
    comp_sizes: dict[int, int] = {}
    for comp_idx, nodes in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True)
    ):
        for node in nodes:
            comp_map[node] = comp_idx
        comp_sizes[comp_idx] = len(nodes)
    return comp_map, comp_sizes


def _component_palette(comp_sizes: dict[int, int]) -> dict[int, str]:
    if not comp_sizes:
        return {}
    num = len(comp_sizes)
    if num <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20").resampled(num)
    else:
        cmap = matplotlib.colormaps.get_cmap("gist_ncar").resampled(num)
    if hasattr(cmap, "colors"):
        color_list = list(cmap.colors)
    else:
        denom = max(num - 1, 1)
        color_list = [cmap(idx / denom) for idx in range(num)]
    palette: dict[int, str] = {}
    for idx, comp_id in enumerate(sorted(comp_sizes, key=lambda cid: (-comp_sizes[cid], cid))):
        palette[comp_id] = to_hex(color_list[idx % len(color_list)])
    return palette


def _graph_metrics(graph: nx.Graph, comp_sizes: dict[int, int]) -> dict[str, float]:
    num_nodes = graph.number_of_nodes()
    degrees = dict(graph.degree())
    metrics: dict[str, float] = {
        "num_nodes": float(num_nodes),
        "num_edges": float(graph.number_of_edges()),
        "num_components": float(len(comp_sizes)),
        "avg_degree": (sum(degrees.values()) / num_nodes) if num_nodes else math.nan,
        "largest_component_nodes": float(max(comp_sizes.values(), default=0)),
    }
    try:
        metrics["avg_local_clustering"] = float(nx.average_clustering(graph))
    except nx.NetworkXError:
        metrics["avg_local_clustering"] = math.nan
    return metrics


def _world_background() -> gpd.GeoDataFrame:
    global WORLD_BACKGROUND, WORLD_BACKGROUND_ERROR
    if WORLD_BACKGROUND is None and not WORLD_BACKGROUND_ERROR:
        try:
            path = gpd.datasets.get_path("naturalearth_lowres")
            world = gpd.read_file(path)
            WORLD_BACKGROUND = world.to_crs(epsg=3857)
        except Exception as exc:  # pragma: no cover - environment specific
            WORLD_BACKGROUND_ERROR = True
            print(
                f"[fallback] Natural Earth dataset unavailable ({exc}); using flat background."
            )
            WORLD_BACKGROUND = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:3857")
    if WORLD_BACKGROUND is None:
        WORLD_BACKGROUND = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:3857")
    return WORLD_BACKGROUND


def _geodataframes(
    graph: nx.Graph,
    city: str,
    comp_map: dict[int, int],
    palette: dict[int, str],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crs = _city_crs(city)
    nodes_rows = []
    edges_rows = []
    for node, data in graph.nodes(data=True):
        if "x" not in data or "y" not in data:
            continue
        nodes_rows.append(
            {
                "node": node,
                "component": comp_map.get(node, -1),
                "degree": graph.degree(node),
                "color": palette.get(comp_map.get(node, -1), "#888888"),
                "geometry": Point(data["x"], data["y"]),
            }
        )

    for u, v, attr in graph.edges(data=True):
        src = graph.nodes[u]
        dst = graph.nodes[v]
        if "x" not in src or "y" not in src or "x" not in dst or "y" not in dst:
            continue
        # Length defaults to Euclidean distance if missing.
        length = attr.get("length")
        if length is None:
            length = math.dist((src["x"], src["y"]), (dst["x"], dst["y"]))
        comp_id = comp_map.get(u, comp_map.get(v, -1))
        edges_rows.append(
            {
                "u": u,
                "v": v,
                "length": length,
                "component": comp_id,
                "color": palette.get(comp_id, "#aaaaaa"),
                "geometry": LineString([(src["x"], src["y"]), (dst["x"], dst["y"]) ]),
            }
        )

    nodes_gdf = gpd.GeoDataFrame(nodes_rows, crs=crs)
    edges_gdf = gpd.GeoDataFrame(edges_rows, crs=crs)
    return nodes_gdf, edges_gdf


def _plot_geographic(
    city: str,
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    metrics: dict[str, float],
    provider_name: str,
    output_path: Path,
    dpi: int,
) -> None:
    if edges_gdf.empty and nodes_gdf.empty:
        print(f"[{city}] No geometry to plot; skipping geographic figure")
        return

    nodes_proj = nodes_gdf.to_crs(epsg=3857)
    edges_proj = edges_gdf.to_crs(epsg=3857)
    bounds = edges_proj.total_bounds if not edges_proj.empty else nodes_proj.total_bounds
    fig, ax = plt.subplots(figsize=(8, 8))

    if not edges_proj.empty:
        edges_proj.plot(ax=ax, color=edges_proj["color"], linewidth=0.8, alpha=0.85)
    if not nodes_proj.empty:
        node_sizes = 5 + 25 * np.sqrt(nodes_proj["degree"].clip(lower=0))
        nodes_proj.plot(
            ax=ax,
            color=nodes_proj["color"],
            markersize=node_sizes,
            alpha=0.9,
            linewidth=0.0,
        )

    margin_x = max(250.0, (bounds[2] - bounds[0]) * 0.05)
    margin_y = max(250.0, (bounds[3] - bounds[1]) * 0.05)
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    _draw_basemap(ax, city, provider_name)

    ax.set_axis_off()
    ax.set_title(f"{city} Road Graph – Geographic Overlay", fontsize=12)
    _annotate_metrics(ax, metrics)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_basemap(ax: plt.Axes, city: str, provider_name: str) -> None:
    global BASEMAP_FAILED
    if provider_name.lower() in {"none", "plain", "off"}:
        _plot_world(ax)
        return
    if not HAS_CONTEXTILY or BASEMAP_FAILED:
        _plot_world(ax)
        return
    try:
        provider = _resolve_provider(provider_name)
        cx.add_basemap(ax, crs="epsg:3857", source=provider, attribution_size=6)
    except Exception as exc:  # pragma: no cover - network / provider failures
        BASEMAP_FAILED = True
        print(
            f"[{city}] Basemap download failed ({exc}); falling back to Natural Earth background."
        )
        _plot_world(ax)


def _plot_world(ax: plt.Axes) -> None:
    world = _world_background()
    if world.empty:
        ax.set_facecolor("#f5f5f5")
        return
    world.plot(ax=ax, color="#f5f5f5", edgecolor="#c9c9c9", linewidth=0.5)


def _resolve_provider(provider_name: str):
    if not HAS_CONTEXTILY:
        raise RuntimeError("Contextily is not installed. Cannot resolve provider.")
    provider = cx.providers
    for token in provider_name.split("."):
        provider = getattr(provider, token)
    return provider


def _plot_topology(
    city: str,
    graph: nx.Graph,
    comp_map: dict[int, int],
    palette: dict[int, str],
    metrics: dict[str, float],
    output_path: Path,
    dpi: int,
    layout_seed: int,
    max_nodes: int,
) -> None:
    if graph.number_of_nodes() == 0:
        print(f"[{city}] Graph empty; skipping topology figure")
        return

    layout_graph, sampled = _layout_graph_for_plot(graph, max_nodes, layout_seed)
    k_value = (
        1.2 / math.sqrt(layout_graph.number_of_nodes())
        if layout_graph.number_of_nodes() > 1
        else 0.1
    )
    pos = nx.spring_layout(
        layout_graph,
        weight="length",
        seed=layout_seed,
        k=k_value,
    )
    degrees = dict(layout_graph.degree())
    node_sizes = [30 + 40 * math.log1p(degrees[node]) for node in layout_graph.nodes()]
    node_colors = [palette.get(comp_map.get(node, -1), "#999999") for node in layout_graph.nodes()]
    edge_colors = [
        palette.get(comp_map.get(u, -1), "#cccccc") for u, v in layout_graph.edges()
    ]

    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(
        layout_graph,
        pos,
        ax=ax,
        edge_color=edge_colors,
        alpha=0.25,
        width=0.5,
    )
    nx.draw_networkx_nodes(
        layout_graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.2,
        edgecolors="#1f1f1f",
        alpha=0.95,
    )
    ax.set_axis_off()
    extra_lines = None
    if sampled and layout_graph.number_of_nodes():
        extra_lines = [
            f"Layout nodes: {layout_graph.number_of_nodes():,}/{graph.number_of_nodes():,}"
        ]
    ax.set_title(f"{city} Road Graph – Topology Layout", fontsize=12)
    _annotate_metrics(ax, metrics, extra_lines)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _layout_graph_for_plot(
    graph: nx.Graph, max_nodes: int, seed: int
) -> tuple[nx.Graph, bool]:
    if graph.number_of_nodes() <= max_nodes or max_nodes <= 0:
        return graph, False
    selected_nodes = _select_nodes_for_layout(graph, max_nodes, seed)
    subgraph = graph.subgraph(selected_nodes).copy()
    return subgraph, True


def _select_nodes_for_layout(graph: nx.Graph, max_nodes: int, seed: int) -> set[int]:
    rng = random.Random(seed)
    selected: set[int] = set()
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    for comp_nodes in components:
        if len(selected) >= max_nodes:
            break
        comp_list = list(comp_nodes)
        remaining = max_nodes - len(selected)
        if len(comp_list) <= remaining:
            selected.update(comp_list)
            continue
        comp_list.sort(key=lambda node: graph.degree(node), reverse=True)
        top_take = max(1, int(remaining * 0.6))
        top_slice = comp_list[:top_take]
        selected.update(top_slice)
        remaining = max_nodes - len(selected)
        if remaining <= 0:
            break
        tail = comp_list[top_take:]
        if tail:
            sample_count = min(remaining, len(tail))
            selected.update(rng.sample(tail, sample_count))
    return selected


def _annotate_metrics(ax: plt.Axes, metrics: dict[str, float], extra_lines: list[str] | None = None) -> None:
    lines = [
        f"Nodes: {int(metrics['num_nodes']):,}",
        f"Edges: {int(metrics['num_edges']):,}",
        f"Components: {int(metrics['num_components'])}",
        f"Largest component nodes: {int(metrics['largest_component_nodes']):,}",
    ]
    avg_deg = metrics.get("avg_degree")
    if avg_deg is not None and not math.isnan(avg_deg):
        lines.append(f"Avg degree: {avg_deg:.2f}")
    clustering = metrics.get("avg_local_clustering")
    if clustering is not None and not math.isnan(clustering):
        lines.append(f"Avg clustering: {clustering:.3f}")
    if extra_lines:
        lines.extend(extra_lines)

    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        color="#1f1f1f",
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "none",
            "pad": 6,
        },
        verticalalignment="bottom",
    )


def process_city(
    csv_path: Path,
    output_dir: Path,
    provider_name: str,
    dpi: int,
    topology_seed: int,
    topology_max_nodes: int,
) -> None:
    city = _city_name(csv_path)
    print(f"[{city}] loading graph from {csv_path.name}")
    graph = load_osm_graph(csv_path)
    comp_map, comp_sizes = _component_maps(graph)
    palette = _component_palette(comp_sizes)
    metrics = _graph_metrics(graph, comp_sizes)
    nodes_gdf, edges_gdf = _geodataframes(graph, city, comp_map, palette)

    output_dir.mkdir(parents=True, exist_ok=True)
    geo_path = output_dir / f"{city}_geo.png"
    topo_path = output_dir / f"{city}_topology.png"

    _plot_geographic(city, nodes_gdf, edges_gdf, metrics, provider_name, geo_path, dpi)
    _plot_topology(
        city,
        graph,
        comp_map,
        palette,
        metrics,
        topo_path,
        dpi,
        topology_seed,
        topology_max_nodes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_paths",
        nargs="*",
        type=Path,
        help="Optional list of specific edge-list CSV files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where PNG files will be written.",
    )
    parser.add_argument(
        "--tile-provider",
        default=DEFAULT_TILE_PROVIDER,
        help=(
            "Contextily provider path (e.g., 'CartoDB.Voyager'). "
            "Use 'none' to skip tile download entirely."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Image DPI for figure exports.",
    )
    parser.add_argument(
        "--topology-seed",
        type=int,
        default=TOPOLOGY_LAYOUT_SEED,
        help="Seed for the spring_layout RNG to keep figures reproducible.",
    )
    parser.add_argument(
        "--topology-max-nodes",
        type=int,
        default=TOPOLOGY_MAX_NODES,
        help="Maximum number of nodes rendered in the topology layout (sampling keeps it responsive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = discover_csv_paths(args.csv_paths)
    if not csv_paths:
        raise SystemExit("No *_Edgelist.csv files found to process.")

    for csv_path in csv_paths:
        process_city(
            csv_path=csv_path,
            output_dir=args.output_dir,
            provider_name=args.tile_provider,
            dpi=args.dpi,
            topology_seed=args.topology_seed,
            topology_max_nodes=args.topology_max_nodes,
        )


if __name__ == "__main__":
    main()
