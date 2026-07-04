"""Adaptive multiplicative-weight attack scheme."""
from __future__ import annotations

import json
import math
import random
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
from .utils import argmax_with_tiebreak, normalize_weights, smooth_weights, weighted_scores


class AdaptiveMWAttack:
    def __init__(self, graph: nx.Graph, coords: Dict[int, tuple], cfg: AttackConfig) -> None:
        self.cfg = cfg
        self.logger = AttackLogger()
        self.graph = graph.copy()
        self.coords = coords
        self.total_nodes = graph.number_of_nodes()
        self.betweenness = BetweennessHelper(
            mode=cfg.betweenness_mode,
            samples=cfg.k_btwn_samples,
            resample_each_step=cfg.btwn_resample_each_step,
            refresh_interval=cfg.btwn_refresh_interval,
            seed=cfg.seed,
        )
        base_weight = 1.0 / len(cfg.features) if cfg.features else 1.0
        self.weights = {name: base_weight for name in cfg.features}
        self.rng = random.Random(cfg.seed)
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

            candidate_nodes: Dict[str, int] = {}
            candidate_gains: Dict[str, float] = {}
            for feature_name, node_scores in normalized.items():
                if not node_scores:
                    continue
                candidate = argmax_with_tiebreak(node_scores, split_raw, degree_raw)
                candidate_nodes[feature_name] = candidate
                candidate_gains[feature_name] = self._candidate_gain(
                    candidate, current_proxy
                )

            self._update_weights(candidate_gains)

            scores = weighted_scores(normalized, self.weights)
            if not scores:
                break
            chosen = argmax_with_tiebreak(scores, split_raw, degree_raw)
            removed_nodes = self._remove_nodes(chosen)
            removed_count = len(removed_nodes)
            self.removed_total += removed_count
            self.attack_sequence.append(chosen)
            self.removed_node_sets.append(removed_nodes)

            increment = robustness_increment(
                current_lcc_ratio, removed_count, self.total_nodes
            )
            self.robustness_q += increment

            current_lcc_ratio = largest_component_ratio(self.graph, self.total_nodes)
            self.lcc_curve.append(current_lcc_ratio)
            removed_ratio = self.removed_total / self.total_nodes if self.total_nodes else 0.0
            self.removed_ratio_curve.append(removed_ratio)
            lcc_size = int(round(current_lcc_ratio * self.total_nodes)) if self.total_nodes else 0

            log_record = {
                "step": step,
                "removed_seed": chosen,
                "removed_nodes": json.dumps(removed_nodes, ensure_ascii=False),
                "num_removed_this_step": removed_count,
                "remaining_nodes": self.graph.number_of_nodes(),
                "remaining_edges": self.graph.number_of_edges(),
                "lcc_size": lcc_size,
                "lcc_ratio": current_lcc_ratio,
                "proxy_value": current_proxy,
                "weights": json.dumps(self.weights),
                "feature_candidates": json.dumps(candidate_nodes),
                "feature_gains": json.dumps(candidate_gains),
                "official_area_prefix": self.robustness_q,
            }
            self.logger.log(log_record)

            step += 1
            if current_lcc_ratio <= self.cfg.stop_ratio:
                break
            if self.graph.number_of_nodes() == 0:
                break

        return self._result()

    def _log_no_attack(self, lcc_ratio: float) -> None:
        proxy_now = proxy_value(
            self.graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
        )
        record = {
            "step": 0,
            "status": "无需攻击",
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

    def _candidate_gain(self, node: int, current_proxy: float) -> float:
        sim_graph = self.graph.copy()
        removal = self._simulate_removal(sim_graph, node)
        sim_graph.remove_nodes_from(removal)
        after = proxy_value(sim_graph, self.cfg.objective, self.total_nodes, self.cfg.alpha)
        if self.cfg.lookahead_h <= 1:
            return current_proxy - after
        rollout_extra = self._rollout_sum(sim_graph, self.cfg.lookahead_h - 1)
        return -(after + rollout_extra)

    def _rollout_sum(self, graph: nx.Graph, horizon: int) -> float:
        if horizon <= 0:
            return 0.0
        sim_graph = graph.copy()
        temp_betw = self.betweenness.clone()
        step = 0
        total = 0.0
        while step < horizon and sim_graph.number_of_nodes() > 0:
            proxy_now = proxy_value(
                sim_graph, self.cfg.objective, self.total_nodes, self.cfg.alpha
            )
            total += proxy_now
            features = compute_features(
                sim_graph, self.cfg.features, temp_betw, step
            )
            normalized = normalize_feature_matrix(features, self.cfg.epsilon)
            degree_raw = features.get("degree", {node: 0.0 for node in sim_graph.nodes})
            split_raw = features.get("split", {node: 0.0 for node in sim_graph.nodes})
            scores = weighted_scores(normalized, self.weights)
            if not scores:
                break
            chosen = argmax_with_tiebreak(scores, split_raw, degree_raw)
            removal = self._simulate_removal(sim_graph, chosen)
            sim_graph.remove_nodes_from(removal)
            step += 1
        return total

    def _remove_nodes(self, node: int) -> List[int]:
        removal = self._actual_removal_nodes(node)
        self.graph.remove_nodes_from(removal)
        return removal

    def _update_weights(self, candidate_gains: Dict[str, float]) -> None:
        updated = {}
        for feature, gain in candidate_gains.items():
            prev = self.weights.get(feature, 1.0 / len(self.weights) if self.weights else 1.0)
            updated[feature] = prev * math.exp(self.cfg.eta * gain)
        normalized = normalize_weights(updated or self.weights)
        if self.cfg.use_smooth:
            self.weights = smooth_weights(self.weights, normalized, self.cfg.lambda_smooth)
        else:
            self.weights = normalized

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
