#!/usr/bin/env python3
"""Generate per-city, per-radius performance curves with robustness values."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from attack_utils import (
    AttackMetadata,
    load_attack_sequence,
    load_city_graph,
    parse_attack_metadata,
    simulate_attack,
)

# Ensure matplotlib cache writes to a safe path (especially in sandboxed envs)
MPL_CACHE = Path(os.environ["MPLCONFIGDIR"])
MPL_CACHE.mkdir(parents=True, exist_ok=True)

CITY_ORDER = [
    "Chengdu",
    "Dalian",
    "Dongguan",
    "Harbin",
    "Qingdao",
    "Quanzhou",
    "Shenyang",
    "Zhengzhou",
]
RADIUS_ORDER = [0.0, 100.0, 300.0, 500.0]
CITY_INDEX = {city: idx for idx, city in enumerate(CITY_ORDER)}
RADIUS_INDEX = {radius: idx for idx, radius in enumerate(RADIUS_ORDER)}

STRATEGY_INFO: Mapping[str, Dict[str, object]] = {
    "随机": {
        "label": "随机打击",
        "color": "#B0B0B0",
        "linestyle": "-",
        "linewidth": 2.5,
        "zorder": 4,
    },
    "度": {
        "label": "动态度贪心",
        "color": "#4A90E2",
        "linestyle": "--",
        "linewidth": 2.5,
        "zorder": 5,
    },
    "k-core": {
        "label": "动态 k-core 贪心",
        "color": "#50E3C2",
        "linestyle": "-.",
        "linewidth": 2.5,
        "zorder": 6,
    },
    "介数": {
        "label": "动态近似介数贪心",
        "color": "#F5A623",
        "linestyle": ":",
        "linewidth": 2.5,
        "zorder": 6,
    },
    "分割收益": {
        "label": "动态分裂增益贪心",
        "color": "#9013FE",
        "linestyle": "--",
        "linewidth": 2.5,
        "zorder": 7,
    },
    "adaptive": {
        "label": "边际收益驱动的自适应调权",
        "color": "#8B572A",
        "linestyle": "-.",
        "linewidth": 2.5,
        "zorder": 9,
    },
    "weights": {
        "label": "权重池搜索的动态调权",
        "color": "#D0021B",
        "linestyle": "-",
        "linewidth": 3.5,
        "zorder": 11,
    },
}

STRATEGY_ORDER: List[str] = ["随机", "度", "k-core", "介数", "分割收益", "adaptive", "weights"]
STRATEGY_INDEX = {name: idx for idx, name in enumerate(STRATEGY_ORDER)}


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "SimSun",
                "Songti SC",
                "SimHei",
                "Microsoft YaHei",
                "Arial Unicode MS",
            ],
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "lines.linewidth": 2.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "figure.figsize": (8, 5),
            "savefig.bbox": "tight",
            "savefig.format": "pdf",
            "legend.frameon": False,
        }
    )


def iter_attack_csvs(targets: Iterable[Path]) -> List[Path]:
    csv_files: List[Path] = []
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(f"Input path not found: {target}")
        if target.is_dir():
            csv_files.extend(sorted(target.glob("*.csv")))
        elif target.suffix.lower() == ".csv":
            csv_files.append(target)
        else:
            raise ValueError(f"Unsupported input: {target}")
    if not csv_files:
        raise ValueError("No CSV files found in the provided inputs.")
    return csv_files


def group_metadata(csv_files: Iterable[Path]) -> Dict[Tuple[str, float], Dict[str, AttackMetadata]]:
    grouped: Dict[Tuple[str, float], Dict[str, AttackMetadata]] = {}
    for csv_path in csv_files:
        meta = parse_attack_metadata(csv_path)
        grouped.setdefault((meta.city, meta.radius), {})[meta.strategy] = meta
    return grouped


def compute_curve(meta: AttackMetadata):
    graph = load_city_graph(meta.graph_csv)
    attack_sequence = load_attack_sequence(meta.csv_path)
    metrics, _, _ = simulate_attack(graph, attack_sequence, meta.radius)
    removed_ratio = np.array([m.removed_ratio for m in metrics], dtype=float)
    lcc_ratio = np.array([m.largest_component_ratio for m in metrics], dtype=float)
    trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None:
        trapz_fn = np.trapezoid
    robustness = float(trapz_fn(lcc_ratio, removed_ratio))
    return removed_ratio, lcc_ratio, robustness


def format_radius(radius: float) -> str:
    if abs(radius - int(radius)) < 1e-9:
        return str(int(radius))
    return f"{radius:.2f}".rstrip("0").rstrip(".")


def plot_city_radius(
    city: str,
    radius: float,
    strategy_to_meta: Dict[str, AttackMetadata],
    output_dir: Path,
    dpi: int = 300,
    summary_rows: List[Dict[str, object]] | None = None,
) -> None:
    available_strategies = sorted(strategy_to_meta)
    if not available_strategies:
        print(f"[WARN] No strategies for {city} radius {radius}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("瘫痪节点比例 f")
    ax.set_ylabel("最大连通分量比例 S")
    ax.set_title(f"{city} | 攻击半径 = {format_radius(radius)}", pad=12)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.3, color="#B0B0B0")
    ax.set_axisbelow(True)

    ordered_strategies = [s for s in STRATEGY_ORDER if s in strategy_to_meta]
    ordered_strategies.extend([s for s in available_strategies if s not in STRATEGY_ORDER])

    for strategy in ordered_strategies:
        meta = strategy_to_meta[strategy]
        removed_ratio, lcc_ratio, robustness = compute_curve(meta)

        info = STRATEGY_INFO.get(strategy, {})
        display = info.get("label", strategy or "未命名策略")
        color = info.get("color", "#333333")
        linestyle = info.get("linestyle", "-")
        linewidth = info.get("linewidth", 2.5)
        zorder = info.get("zorder", 5)

        ax.plot(
            removed_ratio,
            lcc_ratio,
            label=f"{display} (R={robustness:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=zorder,
        )

        if summary_rows is not None:
            summary_rows.append(
                {
                    "city": city,
                    "strategy_key": strategy,
                    "strategy": display,
                    "radius": radius,
                    "robustness": robustness,
                }
            )

    legend = ax.legend(
        loc="upper right",
        fontsize=10,
        labelspacing=0.5,
        handlelength=2.6,
    )
    if legend:
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_facecolor("none")
    sns.despine(ax=ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{city}_radius_{format_radius(radius)}.png"
    output_path = output_dir / file_name
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {output_path}")


def sort_summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def strategy_rank(key: str) -> Tuple[int, str]:
        return (STRATEGY_INDEX.get(key, len(STRATEGY_INDEX)), key)

    return sorted(
        rows,
        key=lambda r: (
            CITY_INDEX.get(r["city"], len(CITY_INDEX)),
            RADIUS_INDEX.get(float(r["radius"]), len(RADIUS_INDEX)),
            strategy_rank(str(r["strategy_key"])),
        ),
    )


def write_summary_csv(rows: List[Dict[str, object]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "robustness_summary.csv"
    sorted_rows = sort_summary_rows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["城市", "攻击策略", "攻击半径", "健壮性"])
        for row in sorted_rows:
            writer.writerow(
                [
                    row["city"],
                    row["strategy"],
                    format_radius(float(row["radius"])),
                    f"{row['robustness']:.6f}",
                ]
            )
    print(f"[OK] Saved summary CSV -> {summary_path}")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["attack sequence"],
        help="CSV files or directories that contain attack sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default="combined_figures",
        help="Directory to store the combined figures (default: combined_figures).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saving outputs (default: 300)",
    )
    args = parser.parse_args()

    configure_style()

    input_paths = [Path(p) for p in args.inputs]
    csv_files = iter_attack_csvs(input_paths)
    grouped = group_metadata(csv_files)
    output_dir = Path(args.output_dir)

    summary_rows: List[Dict[str, object]] = []

    for city in CITY_ORDER:
        for radius in RADIUS_ORDER:
            meta_map = grouped.get((city, float(radius)), {})
            if not meta_map:
                print(f"[WARN] Missing data for {city} radius {radius}")
                continue
            plot_city_radius(
                city,
                float(radius),
                meta_map,
                output_dir,
                dpi=args.dpi,
                summary_rows=summary_rows,
            )

    if summary_rows:
        write_summary_csv(summary_rows, output_dir)


if __name__ == "__main__":
    main()
