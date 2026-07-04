# Dynamic Draw 使用指南

本指南面向对计算机不太熟悉的同学，帮助你一步步运行 `dynamic_draw.py` 并调整几个常用参数。下列命令假定终端当前位于仓库根目录。

## 1. 环境与数据准备

1. **终端定位**：在仓库根目录执行 `cd "附件/Problem3:4/3:4visualize"` 进入可视化目录。
2. **Python 解释器**：脚本依赖你本地安装且包含 `networkx`、`numpy`、`scipy`、`matplotlib` 等包的 Python。记下自己的解释器路径（示例：`/opt/homebrew/.../myenv/bin/python` 或 `~/miniconda3/envs/myenv/bin/python`）。如果不确定，可在激活虚拟环境后运行 `which python` 获取实际路径，后文命令里的 `<PYTHON>` 都替换成这个路径。麻烦的话可以直接把`<PYTHON>`替换为`python`
3. **数据文件**：攻击序列 CSV 存在 `attack sequence/` 目录下，图数据存放在 `data/<城市>_Edgelist.csv`。`dynamic_draw.py` 会根据攻击序列文件名自动匹配城市和攻击半径，因此不用手动指定。
4. **不要删除 `build_osm_graph.py`**：该脚本提供的 `load_osm_graph` 会被 `dynamic_draw.py` 和 `plot_performance_curves.py` 自动调用，用于读取 `data/*.csv`，删除会导致程序运行失败。

## 2. 快速运行

```bash
<PYTHON> dynamic_draw.py "attack sequence/Dalianadaptive500.csv"
```

- 将示例路径替换为你想观察的攻击序列 CSV。
- 程序会先打印“正在预计算帧…”，随后弹出一个包含道路网络和滑块的 Matplotlib 窗口。滑块用于切换帧，查看不同删除步下的最大连通分量。

### Matplotlib 缓存提示

若首次运行出现 Matplotlib 缓存目录不可写的黄色警告，可按以下步骤解决：

```bash
mkdir -p /tmp/mpl
MPLCONFIGDIR=/tmp/mpl <PYTHON> dynamic_draw.py "attack sequence/Dalianadaptive500.csv"
```

这样 Matplotlib 会把缓存写到 `/tmp/mpl`，速度更快且不会再报错。

## 3. 常用参数

所有参数都通过命令行选项传入，可在任意顺序追加到主命令后：

| 参数 | 默认值 | 作用 | 适用场景 |
| --- | --- | --- | --- |
| `--frame-stride` | `200` | 预计算帧之间的间隔步数；越大帧数越少，速度越快但动画跳跃；越小越平滑但耗时更久。 | 大图或长序列可设置 300-500；想看更细腻的变化可设为 50-100。 |
| `--max-frames` | `40` | 限制总帧数。即使 stride 很小，只要帧超过该值就会自动均匀抽样。传入 0 或负数表示不限制。 | 机器内存较小或只需概览趋势时保留默认值；需要完整序列时设为较大数字。 |
| `--node-size` | `1.0` | 控制散点大小。 | 屏幕分辨率高或节点稠密时可调小，稀疏图可调大。 |
| `--edge-width` | `0.2` | 控制边宽。 | 线条太细看不清时调大。 |
| `--title` | `"Dynamic Remaining Graph"` | 自定义图窗标题前缀，程序会自动附加城市、策略和半径。 | 同时打开多座城市窗口时，用标题区分。 |

### 参数示例

```bash
<PYTHON> dynamic_draw.py \
  "attack sequence/Dalianadaptive500.csv" \
  --frame-stride 400 \
  --max-frames 30 \
  --node-size 0.8 \
  --edge-width 0.3 \
  --title "Dalian adaptive"
```

## 4. 进阶：批量性能曲线（可选）

若需要批量输出“largest connected component / N” 对 “removed ratio”的性能曲线，可运行：

```bash
<PYTHON> plot_performance_curves.py \
  --output-dir figures/performance_curves \
  "attack sequence"
```

- 会遍历目录内所有 CSV，模拟带攻击半径的删点过程，生成与 CSV 同名的 PNG。
- `--output-dir` 可换成任意可写目录。

## 5. 故障排查

1. **提示找不到 `networkx` 或其他库**：说明使用到了系统 Python。请确认命令前缀是你记录下来的 `<PYTHON>` 并确保该环境已安装所需依赖。
2. **窗口不显示**：在服务器或无图形界面环境运行时，Matplotlib 无法打开窗口。此时可考虑在本地 Mac 运行，或暂时加 `MPLBACKEND=Agg` 仅验证预计算逻辑，但无法直接交互。
3. **报错找不到图 CSV**：确认攻击序列文件名包含城市前缀（如 `Dalian`、`Chengdu` 等），并确保 `data/` 目录中有对应的 `<城市>_Edgelist.csv`。
4. **运行慢或内存高**：增大 `--frame-stride`、减小 `--max-frames`，或仅选择部分 CSV 运行。
