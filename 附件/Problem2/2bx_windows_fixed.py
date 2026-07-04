"""
批量计算交通网络健壮性 (多进程并行加速版)

功能：
1. 自动读取指定文件夹下的所有 OSM 边列表 CSV 文件。
2. 将 0%~100% 的破坏比例分配给 N 个 CPU 核心并行计算，大幅提速。
3. 自动为每个 CSV 绘制性能曲线并导出图片。
4. 额外导出每个城市的性能曲线 CSV（保存每一步的比例和比值）。
"""
from __future__ import annotations

import csv
import random
import multiprocessing as mp
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ==========================================================
# 0. 全局参数配置区 (你只需要在这里修改参数！)
# ==========================================================
# 默认使用仓库内 Problem1 的八座城市边表；也可替换为其他数据目录。
TARGET_FOLDER = Path(__file__).resolve().parents[1] / "Problem1"

# 设定的 CPU 核心数
N_CPUS = 16

# 步长 (0.01 表示 1%)
STEP = 0.01

# 每个破坏比例独立重复采样的次数
NUM_SIMULATIONS = 50

# ==========================================================

REQUIRED_COLUMNS = ("XCoord", "YCoord", "START_NODE", "END_NODE", "EDGE", "LENGTH")


# ==========================================================
# 0.1 matplotlib 中文字体配置（兼容 Windows / Mac）
# ==========================================================
def configure_matplotlib_for_chinese() -> None:
    """
    为 matplotlib 自动选择可用的中文字体，避免 Windows 上中文变方框。
    优先尝试 Windows 常见字体，其次兼容 Mac / Linux 常见字体。
    """
    preferred_fonts = [
        # Windows 常见中文字体
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "DengXian",
        # macOS 常见中文字体
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        # Linux / 跨平台常见字体
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ]

    available_fonts = {font.name for font in fm.fontManager.ttflist}
    chosen_font = next((font for font in preferred_fonts if font in available_fonts), None)

    if chosen_font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [chosen_font] + [
            font for font in preferred_fonts if font != chosen_font
        ]
        print(f"  > matplotlib 中文字体已设置为: {chosen_font}")
    else:
        print("  > [警告] 未找到常见中文字体，图片中的中文仍可能显示异常。")
        print("  > [提示] Windows 建议安装或启用：Microsoft YaHei / SimHei / SimSun")

    plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 1. 构图模块
# ==========================================================
def load_osm_graph(csv_path: Path) -> nx.Graph:
    """读取 OSM CSV，生成 NetworkX 无向图。"""
    graph = nx.Graph()
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV file {csv_path} is missing required columns: {', '.join(missing)}")

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
                node_data = graph.nodes[start]
                if "x" not in node_data:
                    node_data.update(x=xcoord, y=ycoord)

            if end not in graph:
                graph.add_node(end)

            if graph.has_edge(start, end):
                if length < graph[start][end]["length"]:
                    graph[start][end].update(length=length, edge_id=edge_id)
            else:
                graph.add_edge(start, end, length=length, edge_id=edge_id)

    return graph


# ==========================================================
# 2. 健壮性分析核心代码 (多进程并行版)
# ==========================================================
def get_giant_component_ratio(graph: nx.Graph, original_size: int) -> float:
    """计算当前图中最大连通分量节点数与原图总节点数的比值"""
    if len(graph) == 0:
        return 0.0
    largest_cc = max(nx.connected_components(graph), key=len)
    return len(largest_cc) / original_size


def _worker_simulate_fraction(args):
    """
    多进程的独立工作函数（必须放在顶层供子进程调用）。
    接收指定的破坏比例 frac，独立进行多次随机破坏，返回该比例下的平均连通率。
    """
    G, frac, num_simulations, original_size = args

    if frac == 0.0:
        return frac, get_giant_component_ratio(G, original_size)

    num_to_remove = int(frac * original_size)
    original_nodes = list(G.nodes())
    current_y_list = []

    for _ in range(num_simulations):
        # 随机抽取要破坏的节点
        nodes_to_remove = set(random.sample(original_nodes, num_to_remove))
        # 剩下的完好节点
        nodes_to_keep = [n for n in original_nodes if n not in nodes_to_remove]
        # 直接使用 subgraph 切割出完好网络
        G_temp = G.subgraph(nodes_to_keep)

        y = get_giant_component_ratio(G_temp, original_size)
        current_y_list.append(y)

    true_y = np.mean(current_y_list)
    return frac, true_y


def simulate_random_attacks_parallel(G: nx.Graph, step: float, num_simulations: int, n_cpus: int):
    """分配任务给多核 CPU 并行计算"""
    original_size = G.number_of_nodes()
    fractions = np.round(np.arange(0, 1.0 + step / 2, step), 10)

    print(f"  [多进程启动] 分配 {len(fractions)} 个计算任务给 {n_cpus} 个 CPU 核心...")

    # 准备传给每个 CPU 的参数包
    tasks = [(G, frac, num_simulations, original_size) for frac in fractions]

    results = []
    # 使用 Pool 进程池将任务分发给 n_cpus 个核心
    with mp.Pool(processes=n_cpus) as pool:
        # imap_unordered 会让先算完的进程先返回结果，大幅减少等待时间
        for i, (frac, avg_y) in enumerate(pool.imap_unordered(_worker_simulate_fraction, tasks)):
            results.append((frac, avg_y))
            # 简单打印进度条
            if (i + 1) % 10 == 0:
                print(f"    -> 已计算完成 {i + 1}/{len(fractions)} 个节点破坏比例...")

    # 因为使用无序返回，最后需要按照横坐标 frac 重新排序
    results.sort(key=lambda x: x[0])

    x_fracs = np.array([r[0] for r in results])
    y_ratios = np.array([r[1] for r in results])

    return x_fracs, y_ratios


# ==========================================================
# 2.1 导出性能曲线表格
# ==========================================================
def save_performance_curve_csv(
    folder_path: Path,
    city_name: str,
    x_fracs: np.ndarray,
    y_ratios: np.ndarray,
) -> Path:
    """
    保存每个城市的性能曲线表格：
    - step：第几步
    - damage_fraction：节点破坏比例
    - damage_percent：节点破坏百分比
    - giant_component_ratio：最大连通分量相对规模
    """
    out_csv = folder_path / f"{city_name}_performance_curve.csv"

    # 使用 utf-8-sig，方便 Windows 下直接用 Excel 打开中文不乱码
    with out_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "damage_fraction", "damage_percent", "giant_component_ratio"])

        for idx, (frac, ratio) in enumerate(zip(x_fracs, y_ratios)):
            writer.writerow([idx, float(frac), float(frac * 100), float(ratio)])

    return out_csv


# ==========================================================
# 3. 主程序入口
# ==========================================================
def main() -> None:
    configure_matplotlib_for_chinese()

    folder_path = Path(TARGET_FOLDER)

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"【错误】找不到指定的文件夹：{folder_path}")
        return

    # 抓取该文件夹下所有的 csv 文件
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"【提示】在 {folder_path} 目录下没有找到任何 csv 文件。")
        return

    print(f"在目标文件夹共找到 {len(csv_files)} 个 CSV 文件，开始批量处理...\n")

    # 依次处理每个文件
    for idx, csv_file in enumerate(csv_files):
        city_name = csv_file.stem
        print("=" * 50)
        print(f"【处理进度 {idx + 1}/{len(csv_files)}】 正在分析城市: {city_name}")

        # 1. 读表建图
        print("  > 正在读图建网...")
        graph = load_osm_graph(csv_file)
        print(f"  > 建图完成: {graph.number_of_nodes():,} 个节点, {graph.number_of_edges():,} 条边")

        # 2. 并行模拟计算
        x_fracs, y_ratios = simulate_random_attacks_parallel(
            G=graph,
            step=STEP,
            num_simulations=NUM_SIMULATIONS,
            n_cpus=N_CPUS,
        )

        # 3. 积分与突变点计算
        robustness_score = np.trapz(y_ratios, dx=STEP)
        dy_dx = np.diff(y_ratios) / STEP
        critical_idx = np.argmin(dy_dx)
        critical_fraction = x_fracs[critical_idx]

        print(f"  > 结果: 健壮性积分 R = {robustness_score:.4f}, 临界崩溃点 = {critical_fraction * 100:.0f}%")

        # 4. 保存性能曲线表格
        out_curve_csv = save_performance_curve_csv(folder_path, city_name, x_fracs, y_ratios)
        print(f"  > 性能曲线表格已保存: {out_curve_csv.name}")

        # 5. 绘图保存
        plt.figure(figsize=(8, 6))
        plt.plot(x_fracs, y_ratios, marker=".", markersize=4, color="#1f77b4", linewidth=1.5)
        plt.axvline(
            x=critical_fraction,
            color="red",
            linestyle="--",
            label=f"临界突变点 ({critical_fraction * 100:.0f}%)",
        )
        plt.fill_between(
            x_fracs,
            y_ratios,
            color="#aec7e8",
            alpha=0.4,
            label=f"健壮性积分 $R$ = {robustness_score:.4f}",
        )

        plt.title(f"交通网络健壮性性能曲线 - {city_name}", fontsize=14)
        plt.xlabel("节点遭到随机破坏的比例 $p$", fontsize=12)
        plt.ylabel("最大连通分量相对规模 $P(p)$", fontsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()

        # 将图片保存到 CSV 同一个文件夹里
        out_png = folder_path / f"{city_name}_健壮性曲线.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()  # 必须关闭当前图窗，否则连续处理多个城市时内存会持续增长
        print(f"  > 图片已保存: {out_png.name}")

    print("\n" + "=" * 50)
    print("✅ 所有城市的健壮性分析已全部跑完！图表和曲线表格均已生成。")


# 必须在这里启动 main，否则 Windows / Mac 上的多进程都可能报错
if __name__ == "__main__":
    mp.freeze_support()  # 兼容 Windows 多进程启动
    main()
