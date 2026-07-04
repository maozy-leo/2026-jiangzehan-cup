"""Graph loading utilities for OSM-derived CSV edge lists."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx

REQUIRED_COLUMNS = (
    "XCoord",
    "YCoord",
    "START_NODE",
    "END_NODE",
    "EDGE",
    "LENGTH",
)


def load_graph_from_csv(csv_path: Path) -> nx.Graph:
    """Build an undirected NetworkX graph from an edge-list CSV."""

    graph = nx.Graph()

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            start = int(row["START_NODE"])
            end = int(row["END_NODE"])
            length = float(row["LENGTH"])
            edge_id = int(row["EDGE"])
            xcoord = float(row["XCoord"])
            ycoord = float(row["YCoord"])

            if start not in graph:
                graph.add_node(start, x=xcoord, y=ycoord)
            else:
                data = graph.nodes[start]
                if "x" not in data:
                    data.update(x=xcoord, y=ycoord)

            if end not in graph:
                graph.add_node(end)

            if graph.has_edge(start, end):
                existing = graph[start][end]
                if length < existing["length"]:
                    existing.update(length=length, edge_id=edge_id)
            else:
                graph.add_edge(start, end, length=length, edge_id=edge_id)

    return graph


def extract_coordinates(graph: nx.Graph, coord_mode: str = "xy") -> Dict[int, Tuple[float, float]]:
    """Return node->(x,y) mapping in meters."""

    coord_mode = (coord_mode or "xy").lower()
    if coord_mode == "xy":
        return _extract_xy_coordinates(graph)
    if coord_mode == "lonlat":
        return _extract_lonlat_coordinates(graph)
    raise ValueError(f"Unsupported coord_mode: {coord_mode}")


def _extract_xy_coordinates(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    coords: Dict[int, Tuple[float, float]] = {}
    for node, data in graph.nodes(data=True):
        coords[node] = (float(data.get("x", 0.0)), float(data.get("y", 0.0)))
    return coords


def _extract_lonlat_coordinates(graph: nx.Graph) -> Dict[int, Tuple[float, float]]:
    lonlat: Dict[int, Tuple[float, float]] = {}
    lat_values = []
    for node, data in graph.nodes(data=True):
        lon = _get_first(data, ("lon", "longitude"))
        lat = _get_first(data, ("lat", "latitude"))
        if lon is None or lat is None:
            continue
        lon_f = float(lon)
        lat_f = float(lat)
        lonlat[node] = (lon_f, lat_f)
        lat_values.append(lat_f)

    if not lonlat or not lat_values:
        return _extract_xy_coordinates(graph)

    ref_lat = math.radians(sum(lat_values) / len(lat_values))
    coords: Dict[int, Tuple[float, float]] = {}
    for node, (lon_deg, lat_deg) in lonlat.items():
        coords[node] = _project_lonlat(lon_deg, lat_deg, ref_lat)

    fallback = _extract_xy_coordinates(graph)
    for node in graph.nodes:
        coords.setdefault(node, fallback.get(node, (0.0, 0.0)))
    return coords


def _project_lonlat(lon_deg: float, lat_deg: float, ref_lat_rad: float) -> Tuple[float, float]:
    radius = 6_371_000.0
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = radius * lon_rad * math.cos(ref_lat_rad)
    y = radius * lat_rad
    return x, y


def _get_first(data: dict, keys: Tuple[str, ...]) -> float | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None
