#!/usr/bin/env python3
"""Analyze degree distributions for each city using precomputed stats."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "PingFang SC",
    "Heiti TC",
    "SimHei",
    "Noto Sans CJK SC",
]
plt.rcParams["axes.unicode_minus"] = False

EPSILON = 1e-12
SQRT_TWO = math.sqrt(2.0)
CITY_TABLE_DIR = Path("city_tables")
OUTPUT_DIR = Path("degree_distribution")
SUMMARY_CSV = OUTPUT_DIR / "degree_distribution_summary.csv"
FIT_CSV = OUTPUT_DIR / "degree_distribution_fits.csv"
HIST_FIG = OUTPUT_DIR / "degree_frequency.png"
LOGLOG_FIG = OUTPUT_DIR / "degree_loglog.png"
DEFAULT_SUPPORT_EXTRA = 20
BOOTSTRAP_REPS = 200
CV_FOLDS = 5


@dataclass(slots=True)
class CityDegreeData:
    city: str
    degrees: List[int]
    counts: List[int]

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def min_degree(self) -> int:
        return min(self.degrees, default=0)

    @property
    def max_degree(self) -> int:
        return max(self.degrees, default=0)


@dataclass(slots=True)
class DistributionFit:
    city: str
    name: str
    parameters: Dict[str, float]
    log_likelihood: float
    num_params: int
    cdf_func: Callable[[float], float]
    prob_func: Callable[[int], float]
    support: np.ndarray = field(repr=False)
    pmf: np.ndarray = field(repr=False)
    cvm_stat: float = math.nan
    cvm_pvalue: float = math.nan
    aicc: float = math.nan
    cv_loglik: float = math.nan


def _default_support(data: CityDegreeData, extra: int = DEFAULT_SUPPORT_EXTRA) -> list[int]:
    if not data.degrees:
        return [0]
    start = max(0, data.min_degree)
    span = max(extra, data.max_degree // 2 or 1)
    end = data.max_degree + span
    return list(range(start, end + 1))


def _normalized_pmf(prob_func: Callable[[int], float], support: Sequence[int]) -> np.ndarray:
    raw = np.array([max(prob_func(int(degree)), EPSILON) for degree in support], dtype=float)
    total = raw.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(support), 1.0 / len(support))
    return raw / total


def _expand_samples(data: CityDegreeData) -> np.ndarray:
    if not data.counts:
        return np.array([], dtype=int)
    return np.repeat(np.array(data.degrees, dtype=int), np.array(data.counts, dtype=int))


def _city_data_from_samples(city: str, samples: np.ndarray) -> CityDegreeData:
    if samples.size == 0:
        return CityDegreeData(city=city, degrees=[], counts=[])
    degrees, counts = np.unique(samples.astype(int), return_counts=True)
    return CityDegreeData(city=city, degrees=degrees.tolist(), counts=counts.tolist())


def _cvm_statistic(samples: np.ndarray, cdf_func: Callable[[float], float]) -> float:
    n = samples.size
    if n == 0:
        return float("nan")
    sorted_samples = np.sort(samples)
    total = 0.0
    for idx, value in enumerate(sorted_samples, start=1):
        F = min(max(cdf_func(float(value)), 0.0), 1.0)
        u = (2 * idx - 1) / (2 * n)
        total += (F - u) ** 2
    return total + 1.0 / (12 * n)


def _compute_aicc(loglik: float, num_params: int, sample_size: int) -> float:
    if not math.isfinite(loglik) or sample_size <= 0:
        return float("nan")
    aic = 2 * num_params - 2 * loglik
    denom = sample_size - num_params - 1
    if denom <= 0:
        return float("nan")
    return aic + (2 * num_params * (num_params + 1)) / denom


def _loglik_from_prob_func(samples: np.ndarray, prob_func: Callable[[int], float]) -> float:
    if samples.size == 0:
        return float("nan")
    degrees, counts = np.unique(samples.astype(int), return_counts=True)
    total = 0.0
    for degree, count in zip(degrees, counts):
        prob = max(prob_func(int(degree)), EPSILON)
        total += count * math.log(prob)
    return total


def _bootstrap_cvm(
    city_data: CityDegreeData,
    samples: np.ndarray,
    base_fit: DistributionFit,
    fit_function: Callable[[CityDegreeData], DistributionFit],
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[float, float]:
    if samples.size == 0 or base_fit.support.size == 0:
        return float("nan"), float("nan")
    base_stat = _cvm_statistic(samples, base_fit.cdf_func)
    if not math.isfinite(base_stat):
        return base_stat, float("nan")
    boot_stats = []
    support = base_fit.support
    probs = base_fit.pmf
    n = samples.size
    for _ in range(reps):
        synthetic = rng.choice(support, size=n, p=probs)
        synthetic_data = _city_data_from_samples(city_data.city, synthetic)
        boot_fit = fit_function(synthetic_data)
        boot_samples = _expand_samples(synthetic_data)
        stat = _cvm_statistic(boot_samples, boot_fit.cdf_func)
        if math.isfinite(stat):
            boot_stats.append(stat)
    if not boot_stats:
        return base_stat, float("nan")
    greater = sum(1 for stat in boot_stats if stat >= base_stat)
    pvalue = (greater + 1) / (len(boot_stats) + 1)
    return base_stat, pvalue


def _cross_validated_loglik(
    city_data: CityDegreeData,
    samples: np.ndarray,
    fit_function: Callable[[CityDegreeData], DistributionFit],
    rng: np.random.Generator,
    folds: int = CV_FOLDS,
) -> float:
    n = samples.size
    if n == 0 or folds <= 1 or n < folds:
        return float("nan")
    indices = rng.permutation(n)
    split_indices = np.array_split(indices, folds)
    total_loglik = 0.0
    total_points = 0
    for fold_indices in split_indices:
        if fold_indices.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[fold_indices] = False
        train_samples = samples[mask]
        test_samples = samples[~mask]
        if train_samples.size == 0 or test_samples.size == 0:
            continue
        train_data = _city_data_from_samples(city_data.city, train_samples)
        fit = fit_function(train_data)
        fold_loglik = _loglik_from_prob_func(test_samples, fit.prob_func)
        total_loglik += fold_loglik
        total_points += test_samples.size
    if total_points == 0:
        return float("nan")
    return total_loglik / total_points


def _build_fit(
    city: str,
    name: str,
    parameters: Dict[str, float],
    log_likelihood: float,
    num_params: int,
    prob_func: Callable[[int], float],
    cdf_func: Callable[[float], float],
    support_degrees: Sequence[int],
) -> DistributionFit:
    if not support_degrees:
        support_degrees = [0]
    support_arr = np.array(sorted(set(int(deg) for deg in support_degrees)), dtype=int)
    pmf_arr = _normalized_pmf(prob_func, support_arr)
    return DistributionFit(
        city=city,
        name=name,
        parameters=parameters,
        log_likelihood=log_likelihood,
        num_params=num_params,
        cdf_func=cdf_func,
        prob_func=prob_func,
        support=support_arr,
        pmf=pmf_arr,
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_city_degree_data(path: Path) -> CityDegreeData:
    city = path.stem.replace("_stats", "")
    degrees: List[int] = []
    counts: List[int] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) != 2:
                continue
            metric, value = row
            if metric.startswith("degree_count_"):
                try:
                    degree = int(metric[len("degree_count_"):])
                    count = int(float(value))
                except ValueError:
                    continue
                degrees.append(degree)
                counts.append(count)
    pairs = sorted(zip(degrees, counts), key=lambda item: item[0])
    sorted_degrees = [deg for deg, _ in pairs]
    sorted_counts = [cnt for _, cnt in pairs]
    return CityDegreeData(city=city, degrees=sorted_degrees, counts=sorted_counts)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _weighted_mean(values: Iterable[int], weights: Iterable[int]) -> float:
    num = 0.0
    denom = 0.0
    for value, weight in zip(values, weights):
        num += value * weight
        denom += weight
    return num / denom if denom else float("nan")


def _weighted_moment(values: Iterable[int], weights: Iterable[int], mean: float, order: int) -> float:
    accum = 0.0
    total = 0.0
    for value, weight in zip(values, weights):
        centered = value - mean
        accum += (centered ** order) * weight
        total += weight
    return accum / total if total else float("nan")


def compute_summary_row(data: CityDegreeData) -> Dict[str, float | str]:
    mean = _weighted_mean(data.degrees, data.counts)
    variance = _weighted_moment(data.degrees, data.counts, mean, 2)
    std_dev = math.sqrt(variance) if variance >= 0 else float("nan")
    third_moment = _weighted_moment(data.degrees, data.counts, mean, 3)
    skewness = third_moment / (std_dev ** 3) if std_dev and std_dev > 0 else 0.0
    max_degree = data.max_degree
    max_count = max(data.counts, default=0)
    mode_candidates = [deg for deg, cnt in zip(data.degrees, data.counts) if cnt == max_count]
    mode_value = mode_candidates[0] if mode_candidates else data.min_degree
    return {
        "city": data.city,
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "skewness": skewness,
        "max_degree": max_degree,
        "mode": mode_value,
    }


# ---------------------------------------------------------------------------
# Distribution helper functions
# ---------------------------------------------------------------------------

def _cdf_difference(cdf_func: Callable[[float], float], degree: int) -> float:
    upper = cdf_func(degree + 0.5)
    lower = cdf_func(degree - 0.5)
    prob = upper - lower
    if prob < EPSILON:
        prob = EPSILON
    return prob


def _pareto_cdf(x: float, alpha: float, xmin: float) -> float:
    if x < xmin:
        return 0.0
    if alpha <= 1:
        return 1.0
    return 1.0 - (xmin / x) ** (alpha - 1.0)


def _exponential_cdf(x: float, rate: float, shift: float) -> float:
    if x <= shift:
        return 0.0
    return 1.0 - math.exp(-rate * (x - shift))


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    z = (math.log(x) - mu) / (sigma * SQRT_TWO)
    return 0.5 * (1.0 + math.erf(z))


def _poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else EPSILON
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_cdf(k: int, lam: float) -> float:
    total = 0.0
    for value in range(0, k + 1):
        total += _poisson_pmf(value, lam)
    return min(total, 1.0)


def _negative_binomial_pmf(k: int, r: float, mu: float) -> float:
    if k < 0 or r <= 0 or mu <= 0:
        return 0.0
    log_coeff = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
    log_prob = log_coeff + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu))
    return max(math.exp(log_prob), EPSILON)


def _negative_binomial_cdf(k: int, r: float, mu: float) -> float:
    if not (math.isfinite(r) and math.isfinite(mu)) or r <= 0 or mu <= 0:
        return 1.0 if k >= 0 else 0.0
    total = 0.0
    for value in range(0, k + 1):
        total += _negative_binomial_pmf(value, r, mu)
        if total >= 1.0 - 1e-10:
            return 1.0
    return min(total, 1.0)


# ---------------------------------------------------------------------------
# Distribution fitting implementations
# ---------------------------------------------------------------------------

def fit_power_law(data: CityDegreeData) -> DistributionFit:
    xmin = data.min_degree if data.min_degree > 0 else 1
    denom = 0.0
    for degree, count in zip(data.degrees, data.counts):
        ratio = degree / xmin
        if ratio <= 0:
            continue
        denom += count * math.log(ratio)
    if denom <= 0:
        alpha = float("inf")
    else:
        alpha = 1.0 + data.total / denom
    alpha = max(alpha, 1.0001)

    def continuous_cdf(x: float) -> float:
        return _pareto_cdf(x, alpha, xmin)

    loglik = 0.0
    for degree, count in zip(data.degrees, data.counts):
        prob = _cdf_difference(continuous_cdf, degree)
        loglik += count * math.log(prob)

    prob_func = lambda degree: _cdf_difference(continuous_cdf, degree)
    cdf_func = lambda value: continuous_cdf(value + 0.5)
    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="power_law",
        parameters={"alpha": alpha, "xmin": float(xmin)},
        log_likelihood=loglik,
        num_params=1,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


def fit_exponential(data: CityDegreeData) -> DistributionFit:
    shift = data.min_degree - 0.5
    adjusted_mean_numerator = 0.0
    for degree, count in zip(data.degrees, data.counts):
        adjusted_mean_numerator += (degree - shift) * count
    mean_adjusted = adjusted_mean_numerator / data.total if data.total else float("nan")
    if not mean_adjusted or mean_adjusted <= 0:
        rate = float("nan")
    else:
        rate = 1.0 / mean_adjusted

    def continuous_cdf(x: float) -> float:
        if not math.isfinite(rate) or rate <= 0:
            return 1.0
        return _exponential_cdf(x, rate, shift)

    loglik = 0.0
    for degree, count in zip(data.degrees, data.counts):
        prob = _cdf_difference(continuous_cdf, degree)
        loglik += count * math.log(prob)

    prob_func = lambda degree: _cdf_difference(continuous_cdf, degree)
    cdf_func = lambda value: continuous_cdf(value + 0.5)
    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="exponential",
        parameters={"rate": rate, "shift": shift},
        log_likelihood=loglik,
        num_params=1,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


def fit_lognormal(data: CityDegreeData) -> DistributionFit:
    log_values: List[float] = []
    weights: List[int] = []
    for degree, count in zip(data.degrees, data.counts):
        if degree <= 0 or count <= 0:
            continue
        log_values.append(math.log(degree))
        weights.append(count)
    total_weight = sum(weights)
    if total_weight == 0:
        mu = float("nan")
        sigma = float("nan")
    else:
        mu = sum(val * w for val, w in zip(log_values, weights)) / total_weight
        second = sum(((val - mu) ** 2) * w for val, w in zip(log_values, weights)) / total_weight
        sigma = math.sqrt(max(second, 1e-12))

    def continuous_cdf(x: float) -> float:
        if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
            return 1.0 if x >= 0 else 0.0
        return _lognormal_cdf(x, mu, sigma)

    loglik = 0.0
    for degree, count in zip(data.degrees, data.counts):
        prob = _cdf_difference(continuous_cdf, degree)
        loglik += count * math.log(prob)

    prob_func = lambda degree: _cdf_difference(continuous_cdf, degree)
    cdf_func = lambda value: continuous_cdf(value + 0.5)
    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="lognormal",
        parameters={"mu": mu, "sigma": sigma},
        log_likelihood=loglik,
        num_params=2,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


def fit_poisson(data: CityDegreeData) -> DistributionFit:
    lam = _weighted_mean(data.degrees, data.counts)
    loglik = 0.0
    for degree, count in zip(data.degrees, data.counts):
        prob = max(_poisson_pmf(degree, lam), EPSILON)
        loglik += count * math.log(prob)

    prob_func = lambda degree: _poisson_pmf(degree, lam)

    def cdf_func(value: float) -> float:
        upper = int(math.floor(value + 0.5))
        return _poisson_cdf(upper, lam)

    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="poisson",
        parameters={"lambda": lam},
        log_likelihood=loglik,
        num_params=1,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


def fit_negative_binomial(data: CityDegreeData) -> DistributionFit:
    mu = _weighted_mean(data.degrees, data.counts)
    if not math.isfinite(mu) or mu <= 0:
        mu = float("nan")

    def log_likelihood(r: float) -> float:
        if not math.isfinite(r) or r <= 0 or not math.isfinite(mu) or mu <= 0:
            return -float("inf")
        total = 0.0
        for degree, count in zip(data.degrees, data.counts):
            prob = _negative_binomial_pmf(degree, r, mu)
            total += count * math.log(prob)
        return total

    initial_grid = [10 ** exp for exp in np.linspace(-2, 4, 60)]
    initial_grid.append(1e6)  # approximate Poisson limit
    best_r = None
    best_ll = -float("inf")
    for candidate in initial_grid:
        ll = log_likelihood(candidate)
        if ll > best_ll:
            best_ll = ll
            best_r = candidate

    if best_r is None or not math.isfinite(best_r):
        best_r = 1e6
        best_ll = log_likelihood(best_r)

    step = max(best_r * 0.25, 0.01)
    while step > 1e-4:
        improved = False
        for delta in (-step, 0.0, step):
            candidate = max(1e-4, best_r + delta)
            ll = log_likelihood(candidate)
            if ll > best_ll + 1e-6:
                best_ll = ll
                best_r = candidate
                improved = True
        if not improved:
            step *= 0.5

    prob_func = lambda degree: _negative_binomial_pmf(degree, best_r, mu)

    def cdf_func(value: float) -> float:
        upper = int(math.floor(value + 0.5))
        return _negative_binomial_cdf(upper, best_r, mu)

    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="negative_binomial",
        parameters={"r": best_r, "mu": mu},
        log_likelihood=best_ll,
        num_params=2,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


# ---------------------------------------------------------------------------
# Truncated power-law fitting (grid search + local refinement)
# ---------------------------------------------------------------------------

def _truncated_norm(alpha: float, lambd: float, k_min: int, observed_max: int) -> float:
    total = 0.0
    k = k_min
    limit = observed_max + 200
    while True:
        term = (k ** (-alpha)) * math.exp(-lambd * k)
        total += term
        if k >= limit and term < 1e-12:
            break
        k += 1
        if k > observed_max + 1200:
            break
    return total


def _truncated_loglik(alpha: float, lambd: float, data: CityDegreeData) -> float:
    if alpha <= 1.0 or lambd <= 0:
        return -float("inf")
    norm = _truncated_norm(alpha, lambd, data.min_degree, data.max_degree)
    if norm <= 0 or not math.isfinite(norm):
        return -float("inf")
    base = 0.0
    for degree, count in zip(data.degrees, data.counts):
        base += count * (-alpha * math.log(degree) - lambd * degree)
    return base - data.total * math.log(norm)


def _truncated_pmf(alpha: float, lambd: float, data: CityDegreeData) -> Dict[int, float]:
    norm = _truncated_norm(alpha, lambd, data.min_degree, data.max_degree)
    pmf: Dict[int, float] = {}
    total_prob = 0.0
    k = data.min_degree
    limit = data.max_degree + 200
    while True:
        weight = (k ** (-alpha)) * math.exp(-lambd * k)
        prob = weight / norm
        pmf[k] = prob
        total_prob += prob
        if k >= limit and weight < 1e-12:
            break
        k += 1
        if k > data.max_degree + 1200:
            break
    return pmf


def fit_truncated_power_law(data: CityDegreeData) -> DistributionFit:
    alpha = max(1.1, 1.0 + data.total / max(sum(c * math.log(degree / data.min_degree) if degree > data.min_degree else 0 for degree, c in zip(data.degrees, data.counts)), 1e-6))
    lambd = 0.1
    best_alpha = alpha
    best_lambda = lambd
    best_ll = _truncated_loglik(best_alpha, best_lambda, data)

    # Coarse grid search to find a good starting point
    for alpha_candidate in [1.1 + 0.1 * i for i in range(0, 40)]:
        for lambda_candidate in [0.01 * (1 + i) for i in range(0, 60)]:
            ll = _truncated_loglik(alpha_candidate, lambda_candidate, data)
            if ll > best_ll:
                best_alpha = alpha_candidate
                best_lambda = lambda_candidate
                best_ll = ll

    step_alpha = 0.2
    step_lambda = 0.02
    while step_alpha > 0.005 or step_lambda > 0.001:
        improved = False
        for alpha_offset in (-step_alpha, 0.0, step_alpha):
            for lambda_offset in (-step_lambda, 0.0, step_lambda):
                candidate_alpha = max(1.01, min(6.0, best_alpha + alpha_offset))
                candidate_lambda = max(1e-4, min(2.0, best_lambda + lambda_offset))
                ll = _truncated_loglik(candidate_alpha, candidate_lambda, data)
                if ll > best_ll + 1e-6:
                    best_alpha = candidate_alpha
                    best_lambda = candidate_lambda
                    best_ll = ll
                    improved = True
        if not improved:
            step_alpha *= 0.5
            step_lambda *= 0.5

    norm_bound = data.max_degree + DEFAULT_SUPPORT_EXTRA + 200
    k_min = max(1, data.min_degree)
    norm = _truncated_norm(best_alpha, best_lambda, k_min, norm_bound)

    def prob_func(degree: int) -> float:
        if degree < k_min:
            return 0.0
        weight = (degree ** (-best_alpha)) * math.exp(-best_lambda * degree)
        return max(weight / norm, EPSILON)

    loglik = 0.0
    for degree, count in zip(data.degrees, data.counts):
        prob = prob_func(degree)
        loglik += count * math.log(prob)

    def cdf_func(value: float) -> float:
        upper = int(math.floor(value + 0.5))
        if upper < k_min:
            return 0.0
        total = 0.0
        for degree in range(k_min, upper + 1):
            total += prob_func(degree)
            if total >= 1.0 - 1e-9:
                return 1.0
        return min(total, 1.0)

    support = _default_support(data)
    return _build_fit(
        city=data.city,
        name="truncated_power_law",
        parameters={"alpha": best_alpha, "lambda": best_lambda},
        log_likelihood=loglik,
        num_params=2,
        prob_func=prob_func,
        cdf_func=cdf_func,
        support_degrees=support,
    )


FIT_FUNCTIONS: Dict[str, Callable[[CityDegreeData], DistributionFit]] = {
    "power_law": fit_power_law,
    "exponential": fit_exponential,
    "lognormal": fit_lognormal,
    "poisson": fit_poisson,
    "negative_binomial": fit_negative_binomial,
    "truncated_power_law": fit_truncated_power_law,
}


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib)
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _grid_shape(n_items: int, cols: int = 4) -> tuple[int, int]:
    rows = max(1, math.ceil(n_items / cols))
    return rows, cols


def _plot_frequency_scatter(city_data: List[CityDegreeData], output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    rows, cols = _grid_shape(len(city_data))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for idx, data in enumerate(city_data):
        ax = flat_axes[idx]
        if not data.counts:
            ax.set_axis_off()
            continue
        ax.scatter(data.degrees, data.counts, color="#4C78A8", s=30, label="观测频数")
        ax.set_title(data.city, fontsize=11)
        ax.set_xlabel("度")
        ax.set_ylabel("频数")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)
    for ax in flat_axes[len(city_data) :]:
        ax.set_axis_off()
    fig.suptitle("城市度分布频数散点图", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_loglog(city_data: List[CityDegreeData], output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    rows, cols = _grid_shape(len(city_data))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax in flat_axes:
        ax.set_axis_off()

    for idx, data in enumerate(city_data):
        ax = flat_axes[idx]
        degrees = np.array(data.degrees, dtype=float)
        counts = np.array(data.counts, dtype=float)
        mask = (degrees > 0) & (counts > 0)
        if mask.sum() < 2:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=ax.transAxes)
            continue
        deg = degrees[mask]
        cnt = counts[mask]
        ax.set_axis_on()
        ax.scatter(deg, cnt, color="#F58518", s=30, label="观测值")
        ax.set_xscale("log")
        ax.set_yscale("log")
        log_x = np.log10(deg)
        log_y = np.log10(cnt)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        y_pred = slope * log_x + intercept
        ss_res = float(np.sum((log_y - y_pred) ** 2))
        ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        x_line = np.linspace(log_x.min(), log_x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(10 ** x_line, 10 ** y_line, color="#003f5c", linewidth=1.5, label=f"线性拟合 (R²={r_squared:.2f})")
        ax.set_title(data.city, fontsize=11)
        ax.set_xlabel("度（对数）")
        ax.set_ylabel("频数（对数）")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5, which="both")
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("城市度分布 log-log 图", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_summary_csv(rows: List[Dict[str, float | str]]) -> None:
    fieldnames = ["city", "mean", "variance", "std_dev", "skewness", "max_degree", "mode"]
    _ensure_dir(SUMMARY_CSV.parent)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_fit_csv(fits: List[DistributionFit]) -> None:
    fieldnames = [
        "city",
        "distribution",
        "parameters",
        "log_likelihood",
        "num_params",
        "cvm_stat",
        "cvm_pvalue",
        "aicc",
        "cv_loglik",
    ]
    _ensure_dir(FIT_CSV.parent)
    with FIT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            param_str = "; ".join(f"{key}={_format_float(value)}" for key, value in fit.parameters.items())
            writer.writerow(
                {
                    "city": fit.city,
                    "distribution": fit.name,
                    "parameters": param_str,
                    "log_likelihood": f"{fit.log_likelihood:.4f}" if math.isfinite(fit.log_likelihood) else "nan",
                    "num_params": fit.num_params,
                    "cvm_stat": f"{fit.cvm_stat:.6f}" if math.isfinite(fit.cvm_stat) else "nan",
                    "cvm_pvalue": f"{fit.cvm_pvalue:.6f}" if math.isfinite(fit.cvm_pvalue) else "nan",
                    "aicc": f"{fit.aicc:.4f}" if math.isfinite(fit.aicc) else "nan",
                    "cv_loglik": f"{fit.cv_loglik:.6f}" if math.isfinite(fit.cv_loglik) else "nan",
                }
            )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:
    city_files = sorted(CITY_TABLE_DIR.glob("*_stats.csv"))
    if not city_files:
        raise SystemExit("No city tables found.")
    data_entries = [load_city_degree_data(path) for path in city_files]
    summary_rows = [compute_summary_row(entry) for entry in data_entries]

    rng = np.random.default_rng(20260404)
    fits: List[DistributionFit] = []
    for entry in data_entries:
        samples = _expand_samples(entry)
        for name, fitter in FIT_FUNCTIONS.items():
            fit_rng_seed = int(rng.integers(2**32 - 1))
            fit = fitter(entry)
            boot_rng = np.random.default_rng(fit_rng_seed)
            cv_rng = np.random.default_rng((fit_rng_seed + 1) % (2**32 - 1))
            fit.cvm_stat, fit.cvm_pvalue = _bootstrap_cvm(entry, samples, fit, fitter, boot_rng)
            fit.aicc = _compute_aicc(fit.log_likelihood, fit.num_params, entry.total)
            fit.cv_loglik = _cross_validated_loglik(entry, samples, fitter, cv_rng)
            fits.append(fit)

    write_summary_csv(summary_rows)
    write_fit_csv(fits)
    _plot_frequency_scatter(data_entries, HIST_FIG)
    _plot_loglog(data_entries, LOGLOG_FIG)
    print(f"Summary written to {SUMMARY_CSV}")
    print(f"Fit diagnostics written to {FIT_CSV}")
    print(f"Histogram figure saved to {HIST_FIG}")
    print(f"Log-log figure saved to {LOGLOG_FIG}")


if __name__ == "__main__":
    main()
