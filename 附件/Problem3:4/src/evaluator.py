"""Evaluation helpers for robustness curves."""
from __future__ import annotations

from typing import Iterable, Sequence


def q_from_logs(logs: Sequence[dict]) -> float:
    if not logs:
        return 0.0
    return float(logs[-1].get("official_area_prefix", 0.0))


def stop_step_from_curve(curve: Sequence[float], threshold: float) -> int:
    for idx, val in enumerate(curve):
        if val <= threshold:
            return idx
    return len(curve) - 1
