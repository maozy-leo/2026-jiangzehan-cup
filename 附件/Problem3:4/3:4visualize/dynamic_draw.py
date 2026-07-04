from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Set

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import networkx as nx

from attack_utils import (
    load_attack_sequence,
    load_city_graph,
    parse_attack_metadata,
    simulate_attack,
)

# 优先使用常见中文黑体字体，保证图中文本可读
_CHINESE_FONT_CANDIDATES = [
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
existing_fonts = plt.rcParams.get("font.sans-serif", [])
plt.rcParams["font.sans-serif"] = _CHINESE_FONT_CANDIDATES + existing_fonts
plt.rcParams["axes.unicode_minus"] = False


def build_frame_steps(
    total_steps: int,
    frame_stride: int = 1,
    max_frames: int | None = 200,
    include_last: bool = True,
) -> List[int]:
    if max_frames is not None and max_frames <= 0:
        max_frames = None

    if frame_stride < 1:
        frame_stride = 1

    frame_steps = list(range(0, total_steps + 1, frame_stride))

    if include_last and frame_steps[-1] != total_steps:
        frame_steps.append(total_steps)

    if max_frames is not None and len(frame_steps) > max_frames:
        idxs = []
        for i in range(max_frames):
            pos = round(i * (len(frame_steps) - 1) / max(max_frames - 1, 1))
            idxs.append(pos)
        idxs = sorted(set(idxs))
        frame_steps = [frame_steps[i] for i in idxs]

    return frame_steps


def build_segments(subgraph: nx.Graph):
    segments = []
    for u, v in subgraph.edges():
        x0 = subgraph.nodes[u].get("x", 0.0)
        y0 = subgraph.nodes[u].get("y", 0.0)
        x1 = subgraph.nodes[v].get("x", 0.0)
        y1 = subgraph.nodes[v].get("y", 0.0)
        segments.append([(x0, y0), (x1, y1)])
    return segments


def build_segments_from_nodes(subgraph: nx.Graph, node_set: Set[int]):
    segments = []
    for u, v in subgraph.edges():
        if u in node_set and v in node_set:
            x0 = subgraph.nodes[u].get("x", 0.0)
            y0 = subgraph.nodes[u].get("y", 0.0)
            x1 = subgraph.nodes[v].get("x", 0.0)
            y1 = subgraph.nodes[v].get("y", 0.0)
            segments.append([(x0, y0), (x1, y1)])
    return segments


def build_node_arrays(subgraph: nx.Graph):
    xs = []
    ys = []
    for node in subgraph.nodes():
        xs.append(subgraph.nodes[node].get("x", 0.0))
        ys.append(subgraph.nodes[node].get("y", 0.0))
    return xs, ys


def precompute_frames(
    graph: nx.Graph,
    attack_sequence: Sequence[int],
    radius: float,
    frame_steps: List[int],
) -> tuple[Dict[int, nx.Graph], Dict[int, Set[int]], Dict[int, float]]:
    metrics, graphs_by_step, lcc_nodes_by_step = simulate_attack(
        graph=graph,
        attack_sequence=attack_sequence,
        radius=radius,
        record_steps=frame_steps,
    )
    performance_map = {m.step: m.largest_component_ratio for m in metrics}
    return graphs_by_step, lcc_nodes_by_step, performance_map


def draw_dynamic_graph_matplotlib(
    graph: nx.Graph,
    attack_sequence: Sequence[int],
    radius: float,
    frame_stride: int = 500,
    max_frames: int | None = 20,
    node_size: float = 1.0,
    edge_width: float = 0.2,
    title: str = "Dynamic Remaining Graph",
    enable_slider: bool = True,
    show_plot: bool = True,
    save_frames_dir: Path | None = None,
    save_dpi: int = 200,
) -> None:
    total_steps = len(attack_sequence)
    frame_steps = build_frame_steps(
        total_steps=total_steps,
        frame_stride=frame_stride,
        max_frames=max_frames,
        include_last=True,
    )

    # 固定全局坐标范围
    all_x = [graph.nodes[node].get("x", 0.0) for node in graph.nodes()]
    all_y = [graph.nodes[node].get("y", 0.0) for node in graph.nodes()]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_pad = (x_max - x_min) * 0.02 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.02 if y_max > y_min else 1.0

    x_range = (x_min - x_pad, x_max + x_pad)
    y_range = (y_min - y_pad, y_max + y_pad)

    # 预计算帧
    print("正在预计算帧...")
    subgraphs, lcc_nodes_map, performance_map = precompute_frames(
        graph,
        attack_sequence,
        radius,
        frame_steps,
    )
    print(f"预计算完成，共 {len(frame_steps)} 帧。")

    # 建图窗
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.15 if enable_slider else 0.05)

    # 初始帧
    initial_step = frame_steps[0]
    current_subgraph = subgraphs[initial_step]
    current_lcc_nodes = lcc_nodes_map[initial_step]

    # 全部边
    segments = build_segments(current_subgraph)
    line_collection = LineCollection(
        segments,
        linewidths=edge_width,
        colors="#BBBBBB",
        alpha=0.7,
    )
    ax.add_collection(line_collection)

    # 全部点
    node_x, node_y = build_node_arrays(current_subgraph)
    scatter = ax.scatter(
        node_x,
        node_y,
        s=node_size,
        c="#4C78A8",
        alpha=0.7,
    )

    # 最大连通分量边（高亮）
    lcc_segments = build_segments_from_nodes(current_subgraph, current_lcc_nodes)
    lcc_line_collection = LineCollection(
        lcc_segments,
        linewidths=edge_width * 2.0,
        colors="#D62728",
        alpha=0.95,
    )
    ax.add_collection(lcc_line_collection)

    # 最大连通分量点（高亮）
    lcc_x = [current_subgraph.nodes[node].get("x", 0.0) for node in current_lcc_nodes]
    lcc_y = [current_subgraph.nodes[node].get("y", 0.0) for node in current_lcc_nodes]
    lcc_scatter = ax.scatter(
        lcc_x,
        lcc_y,
        s=max(node_size * 4.0, 4.0),
        c="#D62728",
        alpha=0.95,
        label="Largest Connected Component",
    )

    ratio = performance_map.get(initial_step, 1.0)
    lcc_size = len(current_lcc_nodes)
    title_text = ax.set_title(
        f"{title} | Step={initial_step} | Ratio={ratio:.4f} | LCC={lcc_size}"
    )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    def set_frame_by_step(step: int) -> None:
        subgraph = subgraphs[step]
        lcc_nodes = lcc_nodes_map[step]

        # 更新全部边
        new_segments = build_segments(subgraph)
        line_collection.set_segments(new_segments)
        line_collection.set_linewidth(edge_width)

        # 更新全部点
        new_x, new_y = build_node_arrays(subgraph)
        if len(new_x) == 0:
            scatter.set_offsets([])
        else:
            scatter.set_offsets(list(zip(new_x, new_y)))
            scatter.set_sizes([node_size] * len(new_x))

        # 更新最大连通分量边
        new_lcc_segments = build_segments_from_nodes(subgraph, lcc_nodes)
        lcc_line_collection.set_segments(new_lcc_segments)
        lcc_line_collection.set_linewidth(edge_width * 2.0)

        # 更新最大连通分量点
        new_lcc_x = [subgraph.nodes[node].get("x", 0.0) for node in lcc_nodes]
        new_lcc_y = [subgraph.nodes[node].get("y", 0.0) for node in lcc_nodes]
        if len(new_lcc_x) == 0:
            lcc_scatter.set_offsets([])
        else:
            lcc_scatter.set_offsets(list(zip(new_lcc_x, new_lcc_y)))
            lcc_scatter.set_sizes([max(node_size * 4.0, 4.0)] * len(new_lcc_x))

        ratio = performance_map.get(step, None)
        lcc_size = len(lcc_nodes)
        if ratio is None:
            title_text.set_text(f"{title} | Step={step} | LCC={lcc_size}")
        else:
            title_text.set_text(f"{title} | Step={step} | Ratio={ratio:.4f} | LCC={lcc_size}")

    if enable_slider:
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        step_slider = Slider(
            ax=slider_ax,
            label="Frame Index",
            valmin=0,
            valmax=len(frame_steps) - 1,
            valinit=0,
            valstep=1,
        )

        def on_slider_change(val: float) -> None:
            idx = int(val)
            if idx < 0 or idx >= len(frame_steps):
                return
            set_frame_by_step(frame_steps[idx])
            fig.canvas.draw_idle()

        step_slider.on_changed(on_slider_change)

    if save_frames_dir is not None:
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        for idx, step in enumerate(frame_steps):
            set_frame_by_step(step)
            fig.canvas.draw()
            output_path = save_frames_dir / f"{idx:04d}_step{step}.png"
            fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
        # 恢复初始帧，方便后续交互
        set_frame_by_step(initial_step)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动态展示带攻击半径的删点过程")
    parser.add_argument(
        "attack_csv",
        type=Path,
        help="攻击序列 CSV，例如 attack sequence/Dalianadaptive500.csv",
    )
    parser.add_argument("--frame-stride", type=int, default=200, help="帧之间的步数间隔")
    parser.add_argument("--max-frames", type=int, default=40, help="最大帧数（默认 40）")
    parser.add_argument("--node-size", type=float, default=1.0, help="节点显示大小")
    parser.add_argument("--edge-width", type=float, default=0.2, help="边线宽")
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        default=None,
        help="保存全部帧到指定目录。程序会以 CSV 文件名生成子目录。",
    )
    parser.add_argument(
        "--save-dpi",
        type=int,
        default=200,
        help="保存帧时使用的 DPI（默认 200）。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅生成帧并保存，不弹出交互窗口。",
    )
    parser.add_argument(
        "--disable-slider",
        action="store_true",
        help="禁用窗口底部滑块，适合仅做静态输出。",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Dynamic Remaining Graph",
        help="图窗标题前缀",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = parse_attack_metadata(args.attack_csv)
    graph = load_city_graph(meta.graph_csv)
    attack_sequence = load_attack_sequence(meta.csv_path)

    full_title = f"{args.title} | {meta.city} {meta.strategy or 'attack'} | radius={meta.radius:g}"
    save_dir = None
    if args.save_frames_dir is not None:
        save_dir = args.save_frames_dir / args.attack_csv.stem

    draw_dynamic_graph_matplotlib(
        graph=graph,
        attack_sequence=attack_sequence,
        radius=meta.radius,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        node_size=args.node_size,
        edge_width=args.edge_width,
        title=full_title,
        enable_slider=not args.disable_slider and not args.no_show,
        show_plot=not args.no_show,
        save_frames_dir=save_dir,
        save_dpi=args.save_dpi,
    )


if __name__ == "__main__":
    main()
