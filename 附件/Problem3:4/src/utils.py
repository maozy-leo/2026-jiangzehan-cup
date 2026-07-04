"""Utility helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def argmax_with_tiebreak(
    scores: Dict[int, float],
    split_values: Dict[int, float],
    degree_values: Dict[int, float],
) -> int:
    if not scores:
        raise ValueError("Scores dict is empty")
    max_score = max(scores.values())
    candidates = [node for node, score in scores.items() if score == max_score]
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(
        key=lambda node: (
            -split_values.get(node, 0.0),
            -degree_values.get(node, 0.0),
            node,
        )
    )
    return candidates[0]


def weighted_scores(
    normalized_features: Dict[str, Dict[int, float]],
    weights: Dict[str, float],
) -> Dict[int, float]:
    all_nodes = set()
    for values in normalized_features.values():
        all_nodes.update(values.keys())
    scores = {node: 0.0 for node in all_nodes}
    for feature_name, node_values in normalized_features.items():
        w = weights.get(feature_name, 0.0)
        if w == 0.0:
            continue
        for node, value in node_values.items():
            scores[node] = scores.get(node, 0.0) + w * value
    return scores


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights} if weights else {}
    return {k: v / total for k, v in weights.items()}


def smooth_weights(
    prev: Dict[str, float], new: Dict[str, float], lambda_smooth: float
) -> Dict[str, float]:
    return {
        key: (1 - lambda_smooth) * prev.get(key, 0.0) + lambda_smooth * new.get(key, 0.0)
        for key in prev
    }
