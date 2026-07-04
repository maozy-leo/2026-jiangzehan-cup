"""Feature computation utilities."""
from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple

import networkx as nx

from .tarjan_split import split_gain


class BetweennessHelper:
    """Manage betweenness recomputation policy."""

    def __init__(
        self,
        mode: str = "approx",
        samples: int = 64,
        resample_each_step: bool = True,
        refresh_interval: int = 1,
        seed: int = 2026,
    ) -> None:
        self.mode = mode
        self.samples = samples
        self.resample_each_step = resample_each_step
        self.refresh_interval = max(1, refresh_interval)
        self.seed = seed
        self.cached: Dict[int, float] | None = None
        self.last_refresh_step = -1
        self._rng = random.Random(seed)

    def maybe_refresh(self, graph: nx.Graph, step: int) -> Dict[int, float]:
        needs_refresh = self.cached is None or (step - self.last_refresh_step) >= self.refresh_interval
        if not needs_refresh:
            return self.cached  # type: ignore[return-value]

        seed_value = self._rng.randint(0, 2**31 - 1) if self.resample_each_step else self.seed
        if self.mode == "exact" or graph.number_of_nodes() < max(50, self.samples):
            values = nx.betweenness_centrality(graph, normalized=True, seed=seed_value)
        else:
            k = min(self.samples, graph.number_of_nodes())
            values = nx.betweenness_centrality(graph, k=k, normalized=True, seed=seed_value)
        self.cached = values
        self.last_refresh_step = step
        return values

    def clone(self) -> "BetweennessHelper":
        clone = BetweennessHelper(
            mode=self.mode,
            samples=self.samples,
            resample_each_step=self.resample_each_step,
            refresh_interval=self.refresh_interval,
            seed=self.seed,
        )
        if self.cached is not None:
            clone.cached = dict(self.cached)
        clone.last_refresh_step = self.last_refresh_step
        clone._rng.setstate(self._rng.getstate())
        return clone


def compute_features(
    graph: nx.Graph,
    feature_names: Iterable[str],
    betweenness_helper: BetweennessHelper,
    step: int,
) -> Dict[str, Dict[int, float]]:
    """Compute requested features for all nodes."""

    feature_names = tuple(feature_names)
    features: Dict[str, Dict[int, float]] = {}

    if "degree" in feature_names:
        features["degree"] = {node: float(deg) for node, deg in graph.degree()}

    if "betweenness" in feature_names:
        features["betweenness"] = betweenness_helper.maybe_refresh(graph, step)

    if "kcore" in feature_names:
        core_numbers = nx.core_number(graph) if graph.number_of_nodes() else {}
        features["kcore"] = {node: float(core_numbers.get(node, 0)) for node in graph.nodes}

    if "split" in feature_names:
        gains = split_gain(graph)
        features["split"] = {node: float(gains.get(node, 0)) for node in graph.nodes}

    # ensure every requested feature exists even if graph empty
    for name in feature_names:
        features.setdefault(name, {node: 0.0 for node in graph.nodes})

    return features


def normalize_feature(values: Dict[int, float], epsilon: float) -> Dict[int, float]:
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    if abs(max_v - min_v) <= epsilon:
        return {node: 0.0 for node in values}
    scale = max_v - min_v + epsilon
    return {node: (val - min_v) / scale for node, val in values.items()}


def normalize_feature_matrix(
    features: Dict[str, Dict[int, float]], epsilon: float
) -> Dict[str, Dict[int, float]]:
    return {name: normalize_feature(vals, epsilon) for name, vals in features.items()}
