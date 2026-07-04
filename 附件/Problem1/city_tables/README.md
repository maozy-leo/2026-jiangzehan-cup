# 图统计量计算说明

本 README 对应 `graph_stats.py`，目的是帮助你理解 `city_tables/<city>_stats.csv` 中每一列（尤其是统计量与度计数）的含义。

## 1. 功能与依赖
- **功能**：读取每个 `*_Edgelist.csv`，通过 `build_osm_graph.load_osm_graph` 构建无向图，计算连通性、聚类、路径长度等指标，并输出包含所有标量指标与度计数的 CSV。
- **依赖**：Python 3.10+、NetworkX（其余依赖与 `build_osm_graph.py` 相同）。

## 2. 运行方式
```bash
python graph_stats.py --city-output-dir city_tables
```
可通过尾部传入 `csv_paths` 仅处理某几个城市。

## 3. 输出结构
每个城市会生成一个 `city_tables/<City>_stats.csv`，统一格式：
```
metric,value
num_nodes,18300
...
degree_count_1,2361
...
```
前半部分为单值指标，后半部分为 `degree_count_k`。

## 4. 指标定义
以下指标会按顺序写入 CSV，所有节点/边均来自简化后的无向路网：
1. `num_nodes` / `num_edges`：节点数、边数。
2. `avg_local_clustering`：NetworkX `average_clustering`，表示邻域内部连边占比。
3. `global_transitivity`：全局聚类系数（闭合三角形与连边三元组的比例）。
4. `num_connected_components`：连通分量总数。
5. `network_diameter`：所有连通分量中最大的直径（最远两点之间的最短路径长度）。
6. `average_path_length`：对每个分量的平均最短路径长度按节点对数加权求整体平均；若存在孤立点导致无法计算则为 `nan`。
7. `largest_component_nodes` / `largest_component_edges`：最大连通分量内的节点、边数量。
8. `largest_component_diameter` / `largest_component_avg_path_length`：最大分量内部的直径与平均最短路径长度。
9. `largest_component_size_ratio`：`largest_component_nodes / num_nodes`，衡量主干网络覆盖率。
10. `min_degree` / `max_degree`：全图的最小和最大节点度。
11. 其他指标：若未来在 `metrics` 字典中新增，脚本会自动附加在 CSV 末尾。

## 5. 度计数段
- 从 `degree_count_0`（若存在）到 `degree_count_max_degree`，每行表示“度为 k 的节点数量”。
- 这些计数是 `degree_analysis.py` 的输入，可用于重建原始度分布。
- 加权统计的计算方式示例：平均度 = `sum(k * degree_count_k) / sum(degree_count_k)`。

## 6. 结果解读与联动建议
1. 先看 `largest_component_size_ratio` 判断道路网络是否高度连通（>0.95 表示绝大部分节点处于主干分量）。
2. 对比 `avg_local_clustering` 与 `global_transitivity` 可以揭示城市道路的网格程度，值越大说明存在更多局部回路。
3. 通过 `network_diameter` 与 `average_path_length` 了解道路尺度：若直径、路径长度都大，可能说明道路图跨越更大区域或连通性较弱。
4. 若想解释度分布，可直接查 `degree_count_k` 或将其交由 `degree_analysis.py` 生成更丰富的统计与图像。

掌握上述定义后，读者无需查看源码也能正确理解每一列输出。
