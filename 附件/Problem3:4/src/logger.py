"""Logging and persistence helpers."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from .config import AttackConfig, AttackResult


class AttackLogger:
    def __init__(self) -> None:
        self.logs: List[Dict] = []

    def log(self, record: Dict) -> None:
        self.logs.append(record)

    def as_list(self) -> List[Dict]:
        return self.logs


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_outputs(cfg: AttackConfig, result: AttackResult) -> Path:
    base = Path(cfg.output_dir) / cfg.dataset_name
    _ensure_dir(base)

    # config file for reproducibility
    config_path = base / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg.__dict__, fh, indent=2, ensure_ascii=False)

    # attack sequence csv
    attack_csv = base / "attack_sequence.csv"
    with attack_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "removed_seed", "removed_nodes"])
        for step, (seed_node, removed) in enumerate(
            zip(result.attack_sequence, result.removed_node_sets)
        ):
            writer.writerow([step, seed_node, json.dumps(removed, ensure_ascii=False)])

    # performance curve csv
    perf_csv = base / "performance_curve.csv"
    with perf_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "removed_ratio", "lcc_ratio"])
        for idx, (ratio_removed, lcc_ratio) in enumerate(
            zip(result.removed_ratio_curve, result.lcc_curve)
        ):
            writer.writerow([idx, f"{ratio_removed:.6f}", f"{lcc_ratio:.6f}"])

    # step logs csv (flatten selected keys)
    logs_csv = base / "step_logs.csv"
    if result.logs:
        fieldnames = sorted({key for log in result.logs for key in log})
        with logs_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for log in result.logs:
                writer.writerow(log)

    # summary json
    summary_path = base / "summary.json"
    summary = {
        "dataset_name": cfg.dataset_name,
        "scheme": cfg.scheme,
        "attack_mode": cfg.attack_mode,
        "objective": cfg.objective,
        "robustness_q": result.robustness_q,
        "stop_step": len(result.attack_sequence),
        "final_removed_ratio": result.removed_ratio_curve[-1] if result.removed_ratio_curve else 0.0,
        "seed": cfg.seed,
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return base
