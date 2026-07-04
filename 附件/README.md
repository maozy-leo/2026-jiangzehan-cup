# 数模竞赛附录 README

本 README 汇总当前提交材料中的目录结构、主要脚本职能与常用运行方式，方便评委快速定位代码并复现实验结果。

## 1. 目录速览
| 目录 | 功能定位 | 关键文件/子目录 |
| --- | --- | --- |
| `Problem1/` | OSM 边表清洗、网络构建、基础统计与可视化 | `*_Edgelist.csv` 原始数据、`build_osm_graph.py`、`graph_stats.py`、`degree_analysis.py`、`plot_city_graphs.py`、`city_tables/`、`degree_distribution/`、`figures/` |
| `Problem2/` | 随机攻击基准（多进程模拟 0%~100% 节点破坏） | `2bx_windows_fixed.py` |
| `Problem3:4/` | 问题 3/4 的最优删点策略、批量脚本与可视化工具 | `data/`、`configs/`、`src/`（策略实现）、`scripts/run_all.sh`、`outputs/`、`3:4visualize/` |
| `Problem5/` | 基于介数中心性的修复/加固接口、批处理与性能评估 | `betweenness_profit_interface.py`、`betweenness_repair_interface.py`、`parallel_repair_runner_radius_pool_progress.py`、`parallel_repair_config_radius_pool_progress.json`、`jiaoben3.py` |

## 2. 通用运行环境
- Python ≥ 3.10。
- 必需第三方库：`networkx`、`numpy`、`matplotlib`、`pandas`/`csv`（自带）、`scipy`、`geopandas`、`shapely`、`contextily`（用于底图，可选）、`pyproj`、`multiprocessing`（标准库）、`tqdm`（若需进度条）。
- 推荐步骤：
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows 请使用 .venv\\Scripts\\activate
  pip install -r ../requirements.txt
  ```
- 对于 `Problem3:4/3:4visualize` 的动态图，需要可用的图形界面与 `matplotlib` 的交互后端；无显示环境时可设置 `MPLBACKEND=Agg` 仅生成静态帧。

## 3. Problem1 —— 城市路网构建与统计
### 数据
- `Chengdu_Edgelist.csv` 等 8 座城市的道路边列表，字段 `XCoord,YCoord,START_NODE,END_NODE,EDGE,LENGTH` 已按题面要求清洗。
- `city_tables/Chengdu_stats.csv` 等文件保存节点/边数、连通分量、平均度等预计算指标。

### 核心脚本
| 脚本 | 作用 |
| --- | --- |
| `build_osm_graph.py` | 将单个 `*_Edgelist.csv` 转为 NetworkX 无向图，自动去重平行边并补全节点坐标。示例：`python Problem1/build_osm_graph.py Problem1/Chengdu_Edgelist.csv`。 |
| `graph_stats.py` | 扫描 `city_tables/`，导出更丰富的网络指标 CSV。 |
| `degree_analysis.py` | 读取 `city_tables` 中的度频统计，拟合多种离散分布，输出 `degree_distribution/*.csv/.png` 报告。包括 Kolmogorov–Smirnov / Cramér–von Mises 等检验。 |
| `plot_city_graphs.py` | 同时生成“地理底图 + 拓扑布局”两套 PNG，可选择 Contextily 瓦片或 Natural Earth 备用底图。 |

### 输出约定
- `degree_distribution/`：`degree_distribution_summary.csv`、`degree_distribution_fits.csv`、`degree_frequency.png`、`degree_loglog.png` 总结各城市度分布。
- `figures/`：`plot_city_graphs.py` 生成的地理/拓扑双图，命名规则 `<City>_geo.png` 与 `<City>_topology.png`。

### 快速流程
1. `python Problem1/build_osm_graph.py Problem1/Chengdu_Edgelist.csv` —— 验证原始 CSV 无误。
2. `python Problem1/graph_stats.py --input-dir Problem1/city_tables`（若提供 CLI）或直接使用现有统计表。
3. `python Problem1/degree_analysis.py` —— 自动遍历 `city_tables` 并产出度分布结果。
4. `python Problem1/plot_city_graphs.py --output-dir Problem1/figures` —— 统一渲染八个城市的图像。

## 4. Problem2 —— 多进程随机攻击基线
- `2bx_windows_fixed.py`：
  - 自动读取 `TARGET_FOLDER` 内所有 `*_Edgelist.csv`，构建图后对 0%~100% 的节点随机失效比例做 Monte Carlo 仿真。
  - 支持参数：
    - `TARGET_FOLDER`：数据所在目录（默认为仓库内的 `Problem1/`）。
    - `N_CPUS`：并行进程数，建议设为物理核心数。
    - `STEP`：横轴步长（默认 0.01）。
    - `NUM_SIMULATIONS`：每个破坏比例的独立重复次数。
  - 输出：每个城市都会生成性能曲线 PNG（含中文字体自动配置）以及对应的 CSV（破坏比例 `x` 与最大连通分量比例 `y`）。
  - 运行示例：`python Problem2/2bx_windows_fixed.py`（脚本内部参数即可控制整个流程）。

## 5. Problem3 & Problem4 —— 自适应攻击策略
### 数据与配置
- `data/`：与 Problem1 相同的八座城市边列表，是 `src/` 策略的默认输入。
- `configs/config_adaptive.json` 与 `config_weights.json`：
  - `scheme`：`adaptive_mw`（乘法权重）或 `weight_pool_ls`（权重池 + 局部搜索）。
  - `attack_mode`：`single`（问题 3，只删一点）或 `radius`（问题 4，删除半径 `radius_m` 内的所有节点）。
  - `objective`：`R1`=最大连通分量占比；`R2`=碎裂度复合指标，配合 `alpha`。
  - 其他键控制介数中心性采样、特征集合、lookahead 步数、输出目录等。

### `src/` 模块概览
| 模块 | 职责 |
| --- | --- |
| `runner.py` | 统一 CLI：`python -m src.runner <dataset> <csv> --config configs/config_adaptive.json --radius-m 300`，负责装载图、挑选策略、保存输出。 |
| `config.py` | 定义 `AttackConfig` 与 `AttackResult` 数据类，并给出默认参数。 |
| `graph_loader.py` | 解析 CSV、补齐坐标、可切换 `xy` 与 `lonlat` 模式。 |
| `features.py` | 计算节点特征：度、介数、k-core、Tarjan articulation `split`，并负责归一化。 |
| `adaptive_attack.py` | 问题 3/4 自适应乘法权重算法：逐特征挑选候选节点，按收益更新特征权重，可选 rollout。 |
| `weight_pool_attack.py` | 预定义权重池 + 邻域扰动，循环评估候选权重并选择收益最佳的节点。 |
| `spatial_query.py` | 半径攻击支持：暴力或 KD-tree 查询、节点坐标索引维护。 |
| `proxy_metrics.py` | 统一实现 `R1/R2` 代理目标、碎裂度、累计健壮性积分。 |
| `logger.py` | 将配置、攻击序列、性能曲线、逐步日志保存到 `outputs/<dataset>/`。 |
| `tarjan_split.py` | Tarjan DFS 计算 articulation-based `split_gain`。 |
| `graph_ops.py`、`utils.py`、`evaluator.py` | 图操作、加权合成、处理日志等辅助函数。 |

### 调度与输出
- `scripts/run_all.sh`：批量遍历城市 × 配置 × 半径，自动并发执行 `src.runner`。关键环境变量：`PY_CMD`（Python 可执行文件）和 `MAX_PROCS`（并发度）。
- `outputs/单一指标` 与 `outputs/动态复合指标`：分别对应 `objective=R1` 与 `objective=R2` 的实验记录；下层再按 `problem3/problem4/config_xxx/radius_xxx/城市` 组织。每个城市文件夹包含：
  - `config.json`：完整运行配置。
  - `attack_sequence.csv`：攻击中心与被删节点集合。
  - `performance_curve.csv`：`removed_ratio` vs `lcc_ratio`。
  - `step_logs.csv`：每步日志（候选节点、权重、代理目标等）。
  - `summary.json`：`robustness_q`、停止步数、随机种子等摘要。

### 可视化工具（`3:4visualize/`）
| 组件 | 说明 |
| --- | --- |
| `dynamic_draw.py` | 读取 `attack sequence/*.csv` + `data/*.csv`，交互式播放攻击过程中最大连通分量的变化（带滑块、帧率控制）。
| `plot_performance_curves.py` | 批量把攻击序列转换为性能曲线 PNG。
| `attack_utils.py` | 统一解析文件名中的城市 & 半径、加载图、KD-tree 半径查询、模拟攻击。
| `generate_combined_figures.py`、`combined_figures/` | 将多城市性能曲线拼图展示。
| `figures/`、`dynamic/` | 静态帧与动态图缓存目录。

## 6. Problem5 —— 介数引导的加固与评估
### 核心脚本
| 文件 | 摘要 |
| --- | --- |
| `build_osm_graph.py` | 与 Problem1 相同，用作所有接口的底层图加载器。 |
| `betweenness_profit_interface.py` | “窗口 + 选边收益阈值”策略：
  - `run_betweenness_attack_for_steps` / `run_betweenness_attack_until_target_y`：介数优先攻击并支持半径波及。
  - `run_single_window`：每个窗口先跑基线攻击，再根据收益/新增边比率迭代加边。
  - `run_betweenness_profit_interface`：公开接口，输出攻击序列、加边记录、摘要 JSON。 |
| `betweenness_repair_interface.py` | “阈值跌落”策略：以 `window_drop_slope` 控制攻击窗口内允许的性能下降，逐步添加边直到满足阈值。公开函数 `run_betweenness_repair_interface`（与 profit 版接口兼容）。 |
| `parallel_repair_runner_radius_pool_progress.py` | 批量调度器：读取 JSON 配置，遍历城市 × 半径 × 接口组合，使用 `ProcessPoolExecutor` 调用上述接口并自动绘制性能曲线。 |
| `parallel_repair_config_radius_pool_progress.json` | 示例配置：定义原始数据目录、半径池 `[0,100,300,500]`、两个接口的参数（如 `ratio_gain_threshold`、`window_drop_slope_pool`）、输出目录 `batch_outputs/*`。 |
| `jiaoben3.py` | 并查集 + 逆序恢复评估函数 `compute_robustness_union_find`：给定已生成的攻击序列（含半径扩散），计算性能曲线、累计健壮性 `Q_tau`，可选绘图与 JSON 记录。 |
| `betweenness_profit_interface.py` & `betweenness_repair_interface.py` | 内置半径缓存、KD-tree 最近邻找边、窗口日志等，方便复查。 |

### 批处理流程
1. 根据需要修改 `parallel_repair_config_radius_pool_progress.json`（尤其是 `raw_data_dir`、`sequence_root_dir`、`figure_root_dir` 和接口参数）。
2. 运行：
   ```bash
   python Problem5/parallel_repair_runner_radius_pool_progress.py \
     --config Problem5/parallel_repair_config_radius_pool_progress.json
   ```
3. 结果：
   - `batch_outputs/attack_sequences/<interface>/<City><Strategy><Radius>.csv`
   - `batch_outputs/attack_sequences/<interface>/*_added_edges.csv`
   - `batch_outputs/performance_curves/*.png`
   - 日志中会打印每个任务的 label（城市 | 接口 | radius | slope）。

## 7. 推荐整体工作流
1. **基础校验**：使用 `Problem1/build_osm_graph.py` 验证所有城市 CSV；必要时运行 `graph_stats.py` 和 `plot_city_graphs.py` 获得基础图谱。
2. **随机基线**：运行 `Problem2/2bx_windows_fixed.py`，产出问题 2 的随机攻击曲线，作为后续策略的对照。
3. **优化删点方案**：在 `Problem3:4/` 中选择 `configs/`，通过 `python -m src.runner ...` 或 `scripts/run_all.sh` 跑出问题 3（`--attack-mode single`）与问题 4（`--attack-mode radius --radius-m XXX`）的最优序列。
4. **动画与汇报图**：利用 `3:4visualize/dynamic_draw.py`、`plot_performance_curves.py` 和 `generate_combined_figures.py`，生成 PPT / 附录所需的动态图、性能曲线拼图。
5. **网络加固策略**：在 `Problem5/` 中调用 `run_betweenness_profit_interface` 或 `run_betweenness_repair_interface`，并用 `parallel_repair_runner_radius_pool_progress.py` 批量遍历半径；最后用 `jiaoben3.py` 对生成的攻击序列重新测算健壮性。

## 8. 结果自检清单
- 确认 `outputs/**/summary.json` 中的 `robustness_q` 与正文表格一致。
- `attack_sequence.csv` 与 `performance_curve.csv` 的步数应一一对应，且 `removed_ratio` 单调递增。
- `Problem5` 生成的 `_added_edges.csv` 中新增边数与配置 `add_edge_count_per_round × accepted_rounds` 一致，可追溯 `window_index` 与 `repair_round`。
- 动图使用的攻击序列需与最终提交版本相符，可通过文件名中的城市/策略/半径快速比对。

如需添加新城市，只需在 `Problem1` 与 `Problem3:4/data` 中放入 `<City>_Edgelist.csv`，在 `configs/*.json` 更新 `dataset_name` 与 `graph_path`，并在 `3:4visualize/data` 里同步即可。
