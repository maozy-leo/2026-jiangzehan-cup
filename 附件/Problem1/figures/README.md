# 绘图脚本说明

本文档介绍 `plot_city_graphs.py` 生成的两类 PNG 图像（地理叠加图与拓扑布局图），帮助解释色彩、尺寸与注释含义。

## 1. 功能概述与依赖
- **功能**：遍历所有 `*_Edgelist.csv`，通过 `build_osm_graph.load_osm_graph` 还原道路网络，随后导出两张静态图：
  1. `City_geo.png`：将边/节点投影到 Web Mercator，并叠加底图；
  2. `City_topology.png`：在 spring-layout 平面上展示拓扑结构。
- **依赖**：Python 3.10+、GeoPandas、NetworkX、Matplotlib、pyproj、Shapely。若安装了 Contextily 将自动下载底图；否则退回到 Natural Earth 的矢量轮廓。

## 2. 运行方式
```bash
python plot_city_graphs.py --output-dir figures \
  --tile-provider CartoDB.Voyager \
  --dpi 220 \
  --topology-seed 42 \
  --topology-max-nodes 5000
```
主要参数解释：
- `--output-dir`：PNG 输出目录。
- `--tile-provider`：Contextily 的 provider 路径，填 `none` 可完全跳过底图。
- `--dpi`：导出分辨率，影响文字与线条清晰度。
- `--topology-seed`：spring_layout 随机种子，保证节点位置可复现。
- `--topology-max-nodes`：拓扑图中允许绘制的最大节点数；超限时脚本优先保留高阶节点并对剩余节点采样。
- 传入 `csv_paths` 可只渲染部分城市。

## 3. 地理叠加图 (`*_geo.png`)
- **颜色**：
  - 边线采用各连通分量的专属颜色（最多 20 个分量使用 Matplotlib `tab20`，更多时使用 `gist_ncar`）。
  - 节点颜色与所属分量一致，使断裂区域一目了然。
- **节点大小**：`5 + 25 * sqrt(degree)`，高阶节点的圆点显著更大。
- **透明度**：边 `alpha=0.85`，节点 `alpha=0.9`，便于辨识重叠。
- **底图**：默认调用 `CartoDB.Voyager`；若网络失败或未安装 Contextily，则落到 Natural Earth（米字投影，浅灰背景）。
- **标题/注释**：标题为 “City Road Graph – Geographic Overlay”；左下角白色文本框展示指标（见下节）。

## 4. 拓扑布局图 (`*_topology.png`)
- **布局**：对最多 `topology_max_nodes` 的子图运行 Fruchterman-Reingold (`nx.spring_layout`)，并使用边 `length` 作为权重来缓解过长边。
- **节点形象**：
  - 大小 `30 + 40 * log(1 + degree)`；高度节点不会过分夸张。
  - 边缘描黑 (`edgecolors="#1f1f1f"`) 以增加轮廓。
- **颜色**：同样按连通分量着色；边的颜色继承源节点的色值并降低透明度 (`alpha=0.25`) 以突出节点簇。
- **采样提示**：若发生节点采样，注释框会额外显示 `Layout nodes: X/Y`。
- **背景**：关闭坐标轴，仅展示网络。

## 5. 图中文字指标解释
两张图左下角的文本框由 `_annotate_metrics` 生成，字段含义：
- `Nodes`：节点总数（子图采样前）。
- `Edges`：边总数。
- `Components`：连通分量数量。
- `Largest component nodes`：最大连通分量的节点数。
- `Avg degree`：平均度（若可计算）。
- `Avg clustering`：NetworkX 平均局部聚类系数。
- `Layout nodes`：仅在拓扑图出现，表示绘制的节点数与真实节点数。

## 6. 常见疑问
- **底图变灰/缺失**：多数情况下是 Contextily 获取瓦片失败；脚本会打印 “[City] Basemap download failed (...)” 并自动回退，无需手动干预。
- **色彩与图例**：连通分量颜色没有单独图例，因为颜色仅用于区分区域；若需要固定色板可在 `_component_palette` 中设定。
- **节点过密**：调低 `--topology-max-nodes` 可显著提升可读性；或增大 `--dpi` 并后期缩放。

借助该 README，可直接根据色彩、尺寸与注释解读 PNG 的空间结构与拓扑结构，无需查阅源码。
