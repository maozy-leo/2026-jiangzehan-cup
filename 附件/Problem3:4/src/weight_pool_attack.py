"""Weight-pool based attack scheme."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from .config import AttackConfig, AttackResult
from .features import BetweennessHelper, compute_features, normalize_feature_matrix
from .logger import AttackLogger
from .proxy_metrics import (
    largest_component_ratio,
    proxy_value,
    robustness_increment,
)
from .spatial_query import SpatialIndex, brute_force_nodes_within_radius
from .utils import argmax_with_tiebreak, normalize_weights, weighted_scores


@dataclass
class WeightCandidate:
    name: str
    weights: Dict[str, float]


class WeightPoolAttack:
    def __init__(self, graph: nx.Graph, coords: Dict[int, tuple], cfg: AttackConfig) -> None:
        self.cfg = cfg
        self.graph = graph.copy()
        self.coords = coords
        self.total_nodes = graph.number_of_nodes()
        self.logger = AttackLogger()
        self.betweenness = BetweennessHelper(
            mode=cfg.betweenness_mode,
            samples=cfg.k_btwn_samples,
            resample_each_step=cfg.btwn_resample_each_step,
            refresh_interval=cfg.btwn_refresh_interval,
            seed=cfg.seed,
        )
        self.base_pool = self._build_weight_pool(list(cfg.features))
        self.radius = self._resolve_radius()
        self.spatial_index = SpatialIndex(
            coords,
            method=cfg.spatial_index,
            rebuild=cfg.rebuild_spatial_index,
        )

        self.attack_sequence: List[int] = []
        self.removed_node_sets: List[List[int]] = []
        self.lcc_curve: List[float] = []
        self.removed_ratio_curve: List[float] = []
        self.robustness_q = 0.0
        self.removed_total = 0

    def _build_weight_pool(self, features: List[str]) -> List[WeightCandidate]:
        pool: List[WeightCandidate] = []
        p = len(features)
        if p == 0:
            return [WeightCandidate("uniform", {"degree": 1.0})]

        beta1 = self.cfg.single_priority
        beta2 = self.cfg.dual_priority
        q1 = (1 - beta1) / (p - 1) if p > 1 else 0.0
        q2 = (1 - 2 * beta2) / (p - 2) if p > 2 else 0.0

        for idx, feat in enumerate(features):
            weights = {name: q1 for name in features}
            weights[feat] = beta1
            pool.append(WeightCandidate(f"single_{feat}", weights))

        if p >= 2:
            for i in range(p):
                for j in range(i + 1, p):
                    weights = {name: q2 for name in features}
                    weights[features[i]] = beta2
                    weights[features[j]] = beta2
                    pool.append(WeightCandidate(f"dual_{features[i]}_{features[j]}", weights))

        avg_weight = {name: 1.0 / p for name in features}
        pool.append(WeightCandidate("uniform", avg_weight))
        return pool

    def run(self) -> AttackResult:
        if self.total_nodes == 0:
            return self._result()

        current_lcc_ratio = largest_component_ratio(self.graph, self.total_nodes)
        self.lcc_curve.append(current_lcc_ratio)
        self.removed_ratio_curve.append(0.0)
        if current_lcc_ratio <= self.cfg.stop_ratio:
            self._log_no_attack(current_lcc_ratio)
            return self._result()

        step = 0
        while self.graph.number_of_nodes() > 0:
            self.spatial_index.update_active_nodes(self.graph.nodes)
            if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                break

            current_proxy = proxy_value(
                self.graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
            )
            features = compute_features(
                self.graph,
                self.cfg.features,
                self.betweenness,
                step,
            )
            normalized = normalize_feature_matrix(features, self.cfg.epsilon)
            degree_raw = features.get("degree", {node: 0.0 for node in self.graph.nodes})
            split_raw = features.get("split", {node: 0.0 for node in self.graph.nodes})

            evaluations = self._evaluate_pool(normalized, degree_raw, split_raw, current_proxy)
            if not evaluations:
                break
            best_eval = max(evaluations, key=lambda item: item["gain"])

            chosen_node = best_eval["node"]
            removed_nodes = self._remove_nodes(chosen_node)
            removed_count = len(removed_nodes)
            self.attack_sequence.append(chosen_node)
            self.removed_node_sets.append(removed_nodes)
            self.removed_total += removed_count

            increment = robustness_increment(current_lcc_ratio, removed_count, self.total_nodes)
            self.robustness_q += increment

            current_lcc_ratio = largest_component_ratio(self.graph, self.total_nodes)
            self.lcc_curve.append(current_lcc_ratio)
            removed_ratio = self.removed_total / self.total_nodes if self.total_nodes else 0.0
            self.removed_ratio_curve.append(removed_ratio)

            lcc_size = int(round(current_lcc_ratio * self.total_nodes)) if self.total_nodes else 0
            log_record = {
                "step": step,
                "weights_name": best_eval["name"],
                "weights": json.dumps(best_eval["weights"]),
                "removed_seed": chosen_node,
                "removed_nodes": json.dumps(removed_nodes, ensure_ascii=False),
                "num_removed_this_step": removed_count,
                "remaining_nodes": self.graph.number_of_nodes(),
                "remaining_edges": self.graph.number_of_edges(),
                "lcc_size": lcc_size,
                "lcc_ratio": current_lcc_ratio,
                "proxy_value": current_proxy,
                "feature_gains": json.dumps({e["name"]: e["gain"] for e in evaluations}),
                "official_area_prefix": self.robustness_q,
            }
            self.logger.log(log_record)

            step += 1
            if current_lcc_ratio <= self.cfg.stop_ratio:
                break

        return self._result()

    def _evaluate_pool(
        self,
        normalized: Dict[str, Dict[int, float]],
        degree_raw: Dict[int, float],
        split_raw: Dict[int, float],
        current_proxy: float,
    ) -> List[Dict]:
        results: List[Dict] = []
        seen = set()
        for candidate in self.base_pool:
            normalized_weights = normalize_weights(candidate.weights)
            key = tuple(round(normalized_weights[name], 5) for name in sorted(normalized_weights))
            if key in seen:
                continue
            seen.add(key)
            scores = weighted_scores(normalized, normalized_weights)
            if not scores:
                continue
            node = argmax_with_tiebreak(scores, split_raw, degree_raw)
            gain = self._candidate_gain(node, current_proxy, normalized_weights)
            results.append(
                {
                    "name": candidate.name,
                    "weights": normalized_weights,
                    "node": node,
                    "gain": gain,
                }
            )
        if self.cfg.enable_local_neighbors and results:
            neighbors = self._local_neighbors(results)
            for neighbor in neighbors:
                normalized_weights = neighbor["weights"]
                key = tuple(
                    round(normalized_weights[name], 5) for name in sorted(normalized_weights)
                )
                if key in seen:
                    continue
                seen.add(key)
                scores = weighted_scores(normalized, normalized_weights)
                if not scores:
                    continue
                node = argmax_with_tiebreak(scores, split_raw, degree_raw)
                gain = self._candidate_gain(node, current_proxy, normalized_weights)
                results.append(
                    {
                        "name": neighbor["name"],
                        "weights": normalized_weights,
                        "node": node,
                        "gain": gain,
                    }
                )
        return results

    def _log_no_attack(self, lcc_ratio: float) -> None:
        proxy_now = proxy_value(
            self.graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
        )
        record = {
            "step": 0,
            "weights_name": None,
            "weights": None,
            "removed_seed": None,
            "removed_nodes": "[]",
            "num_removed_this_step": 0,
            "remaining_nodes": self.graph.number_of_nodes(),
            "remaining_edges": self.graph.number_of_edges(),
            "lcc_size": int(round(lcc_ratio * self.total_nodes)) if self.total_nodes else 0,
            "lcc_ratio": lcc_ratio,
            "proxy_value": proxy_now,
            "official_area_prefix": self.robustness_q,
        }
        self.logger.log(record)

    def _local_neighbors(self, evaluated: List[Dict]) -> List[Dict]:
        best = max(evaluated, key=lambda item: item["gain"])
        base_weights = best["weights"]
        delta = self.cfg.local_delta
        features = list(base_weights.keys())
        neighbors: List[Dict] = []
        for feat in features:
            new_weights = base_weights.copy()
            new_weights[feat] = min(1.0, new_weights[feat] + delta)
            remainder = max(0.0, 1.0 - new_weights[feat])
            if len(features) > 1:
                per = remainder / (len(features) - 1)
                for other in features:
                    if other == feat:
                        continue
                    new_weights[other] = max(0.0, per)
            norm = normalize_weights(new_weights)
            neighbors.append({"name": f"local_{feat}", "weights": norm})
        return neighbors

    def _candidate_gain(self, node: int, current_proxy: float, weights: Dict[str, float]) -> float:
        sim_graph = self.graph.copy()
        removal = self._simulate_removal(sim_graph, node)
        sim_graph.remove_nodes_from(removal)
        after = proxy_value(
            sim_graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
        )
        if self.cfg.lookahead_h <= 1:
            return current_proxy - after
        rollout_extra = self._rollout_sum(sim_graph, self.cfg.lookahead_h - 1, weights)
        return -(after + rollout_extra)

    def _rollout_sum(self, graph: nx.Graph, horizon: int, weights: Dict[str, float]) -> float:
        if horizon <= 0:
            return 0.0
        temp_graph = graph.copy()
        temp_betw = self.betweenness.clone()
        total = 0.0
        step = 0
        while step < horizon and temp_graph.number_of_nodes() > 0:
            proxy_now = proxy_value(
                temp_graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
            )
            total += proxy_now
            features = compute_features(
                temp_graph, self.cfg.features, temp_betw, step
            )
            normalized = normalize_feature_matrix(features, self.cfg.epsilon)
            degree_raw = features.get("degree", {node: 0.0 for node in temp_graph.nodes})
            split_raw = features.get("split", {node: 0.0 for node in temp_graph.nodes})
            scores = weighted_scores(normalized, weights)
            if not scores:
                break
            chosen = argmax_with_tiebreak(scores, split_raw, degree_raw)
            removal = self._simulate_removal(temp_graph, chosen)
            temp_graph.remove_nodes_from(removal)
            step += 1
        return total

    def _remove_nodes(self, node: int) -> List[int]:
        removal = self._actual_removal_nodes(node)
        self.graph.remove_nodes_from(removal)
        return removal

    def _actual_removal_nodes(self, node: int) -> List[int]:
        if self.radius <= 0:
            return [node]
        return self.spatial_index.nodes_within_radius(node, self.radius)

    def _simulate_removal(self, graph: nx.Graph, node: int) -> List[int]:
        if self.radius <= 0:
            return [node]
        return brute_force_nodes_within_radius(node, self.radius, self.coords, graph.nodes)

    def _resolve_radius(self) -> float:
        mode = (self.cfg.attack_mode or "single").lower()
        if mode == "single":
            return 0.0
        if mode == "radius":
            if self.cfg.radius_m <= 0:
                raise ValueError("radius attack_mode requires radius_m > 0")
            return self.cfg.radius_m
        raise ValueError(f"Unsupported attack_mode: {self.cfg.attack_mode}")

    def _result(self) -> AttackResult:
        return AttackResult(
            attack_sequence=self.attack_sequence,
            removed_node_sets=self.removed_node_sets,
            lcc_curve=self.lcc_curve,
            removed_ratio_curve=self.removed_ratio_curve,
            robustness_q=self.robustness_q,
            logs=self.logger.as_list(),
        )
