from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt

from attack_utils import (
    AttackStepMetrics,
    load_attack_sequence,
    load_city_graph,
    parse_attack_metadata,
    simulate_attack,
)

DEFAULT_INPUT_DIR = "attack sequence"
DEFAULT_OUTPUT_DIR = Path("figures/performance_curves")


def iter_attack_csvs(targets: Iterable[str]) -> List[Path]:
    csv_files: List[Path] = []
    for target in targets:
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {target}")
        if path.is_dir():
            csv_files.extend(sorted(path.glob("*.csv")))
        elif path.suffix.lower() == ".csv":
            csv_files.append(path)
        else:
            raise ValueError(f"Unsupported input (expect CSV file or directory): {target}")
    if not csv_files:
        raise ValueError("No CSV files found in the provided inputs.")
    return csv_files


def plot_curve(
    metrics: List[AttackStepMetrics],
    output_path: Path,
    subtitle: str,
) -> None:
    removed_ratio = [m.removed_ratio for m in metrics]
    lcc_ratio = [m.largest_component_ratio for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(removed_ratio, lcc_ratio, color="#1f77b4", linewidth=1.8)
    ax.set_title("performance curve")
    ax.set_xlabel("removed ratio")
    ax.set_ylabel("largest connected component / N")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.text(
        0.02,
        0.04,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        alpha=0.8,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute performance curves for attack sequences (with attack radius) "
            "and save a PNG per CSV."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[DEFAULT_INPUT_DIR],
        help="CSV files or directories containing attack sequences (default: 'attack sequence').",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where PNGs will be saved (default: figures/performance_curves).",
    )
    args = parser.parse_args()

    csv_files = iter_attack_csvs(args.inputs)
    output_dir = Path(args.output_dir)

    for csv_file in csv_files:
        print(f"Processing {csv_file} ...")
        meta = parse_attack_metadata(csv_file)
        graph = load_city_graph(meta.graph_csv)
        attack_sequence = load_attack_sequence(csv_file)
        metrics, _, _ = simulate_attack(graph, attack_sequence, meta.radius)

        subtitle = f"{meta.city} {meta.strategy or 'attack'} | radius = {meta.radius:g}"
        output_path = output_dir / f"{csv_file.stem}.png"
        plot_curve(metrics, output_path, subtitle)
        print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
