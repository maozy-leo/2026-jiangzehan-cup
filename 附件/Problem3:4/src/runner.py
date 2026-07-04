"""CLI runner for traffic network robustness experiments."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

from .adaptive_attack import AdaptiveMWAttack
from .config import AttackConfig
from .graph_loader import extract_coordinates, load_graph_from_csv
from .logger import save_outputs
from .weight_pool_attack import WeightPoolAttack


SCHEMES = {
    "adaptive_mw": AdaptiveMWAttack,
    "weight_pool_ls": WeightPoolAttack,
}


def run(cfg: AttackConfig) -> None:
    csv_path = Path(cfg.graph_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    graph = load_graph_from_csv(csv_path)
    coords = extract_coordinates(graph, cfg.coord_mode)
    scheme_cls = SCHEMES.get(cfg.scheme)
    if scheme_cls is None:
        raise ValueError(f"Unsupported scheme: {cfg.scheme}")
    attack = scheme_cls(graph, coords, cfg)
    result = attack.run()
    output_dir = save_outputs(cfg, result)
    print(f"Finished {cfg.dataset_name} -> {output_dir}")


def parse_args() -> AttackConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="Dataset name used for outputs")
    parser.add_argument("graph_path", help="Path to CSV edge list")
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--attack-mode", dest="attack_mode", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--stop-ratio", dest="stop_ratio", type=float, default=None)
    parser.add_argument("--radius-m", dest="radius_m", type=float, default=None)
    parser.add_argument("--lookahead", dest="lookahead_h", type=int, default=None)
    parser.add_argument("--betweenness-mode", dest="betweenness_mode", default=None)
    parser.add_argument("--betweenness-samples", dest="k_btwn_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--config", help="Optional JSON config file")

    args = parser.parse_args()
    defaults = asdict(AttackConfig(dataset_name=args.dataset_name, graph_path=args.graph_path))
    valid_fields = {f.name for f in fields(AttackConfig)}
    cfg_dict = dict(defaults)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for key, value in loaded.items():
            if key in valid_fields:
                cfg_dict[key] = value

    cli_overrides = {
        "dataset_name": args.dataset_name,
        "graph_path": args.graph_path,
        "scheme": args.scheme,
        "attack_mode": args.attack_mode,
        "objective": args.objective,
        "alpha": args.alpha,
        "stop_ratio": args.stop_ratio,
        "radius_m": args.radius_m,
        "lookahead_h": args.lookahead_h,
        "betweenness_mode": args.betweenness_mode,
        "k_btwn_samples": args.k_btwn_samples,
        "seed": args.seed,
        "output_dir": args.output_dir,
    }

    for key, value in cli_overrides.items():
        if value is not None and key in valid_fields:
            cfg_dict[key] = value

    return AttackConfig(**cfg_dict)


if __name__ == "__main__":
    run(parse_args())
