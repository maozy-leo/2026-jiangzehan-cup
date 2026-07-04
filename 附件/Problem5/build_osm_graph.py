"""构建一个基于 OSM 边列表 CSV 的 NetworkX 无向图。

CSV 需包含以下字段：
XCoord,YCoord,START_NODE,END_NODE,EDGE,LENGTH

示例：
    python build_osm_graph.py Chengdu_Edgelist.csv

脚本会打印生成图的节点/边统计信息。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import networkx as nx

REQUIRED_COLUMNS = (
    "XCoord",
    "YCoord",
    "START_NODE",
    "END_NODE",
    "EDGE",
    "LENGTH",
)


def load_osm_graph(csv_path: Path) -> nx.Graph:
    """读取 OSM CSV，生成 NetworkX 无向图。

    每条记录都代表一条有向边，但由于道路是双向的，我们将其折叠为无向边；
    如果同一对节点间存在多条边，则保留长度最短的那条。
    """

    graph = nx.Graph()

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)  # 逐行按字段名读取

        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            # CSV 中的 XCoord/YCoord 是起点 START_NODE 的坐标
            start = int(row["START_NODE"])
            end = int(row["END_NODE"])
            length = float(row["LENGTH"])
            edge_id = int(row["EDGE"])
            xcoord = float(row["XCoord"])
            ycoord = float(row["YCoord"])

            # 将起点的坐标写入节点属性；如果之前未录入，则补写
            if start not in graph:
                graph.add_node(start, x=xcoord, y=ycoord)
            else:
                node_data = graph.nodes[start]
                if "x" not in node_data:
                    node_data.update(x=xcoord, y=ycoord)

            if end not in graph:
                # CSV 中无终点坐标，先创建空节点，坐标等信息等之后其它记录补写
                graph.add_node(end)

            if graph.has_edge(start, end):
                # 多条平行边，保留长度最短的一条
                if length < graph[start][end]["length"]:
                    graph[start][end].update(length=length, edge_id=edge_id)
            else:
                graph.add_edge(start, end, length=length, edge_id=edge_id)

    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the OSM edge-list CSV (e.g., Chengdu_Edgelist.csv)",
    )
    args = parser.parse_args()

    graph = load_osm_graph(args.csv_path)
    print(
        "图构建完成：nodes={:,}, edges={:,}".format(
            graph.number_of_nodes(), graph.number_of_edges()
        )
    )


if __name__ == "__main__":
    main()
