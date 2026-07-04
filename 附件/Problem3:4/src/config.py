"""Configuration objects for robustness experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AttackConfig:
    dataset_name: str
    graph_path: str
    scheme: str = "adaptive_mw"
    attack_mode: str = "single"
    stop_ratio: float = 0.01
    objective: str = "R1"
    alpha: float = 0.7
    features: Tuple[str, ...] = (
        "degree",
        "betweenness",
        "kcore",
        "split",
    )
    betweenness_mode: str = "approx"
    k_btwn_samples: int = 32
    btwn_resample_each_step: bool = True
    btwn_refresh_interval: int = 1
    eta: float = 8.0
    use_smooth: bool = True
    lambda_smooth: float = 0.3
    lookahead_h: int = 1
    epsilon: float = 1e-12
    single_priority: float = 0.55
    dual_priority: float = 0.35
    enable_local_neighbors: bool = True
    local_delta: float = 0.05
    radius_m: float = 0.0
    coord_mode: str = "xy"
    spatial_index: str = "bruteforce"
    rebuild_spatial_index: bool = False
    seed: int = 2026
    max_steps: Optional[int] = None
    verbose: bool = True
    output_dir: str = "./outputs"


@dataclass
class AttackResult:
    attack_sequence: list
    removed_node_sets: list
    lcc_curve: list
    removed_ratio_curve: list
    robustness_q: float
    logs: list
