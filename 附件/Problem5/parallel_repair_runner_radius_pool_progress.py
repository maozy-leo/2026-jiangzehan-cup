from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def _load_module_from_file(module_file: str | Path):
    module_path = Path(module_file).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    module_name = module_path.stem + "_" + str(abs(hash(str(module_path))))
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from file: {module_path}")

    module = importlib.util.module_from_spec(spec)
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_path(base_dir: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _discover_city_graphs(raw_data_dir: Path, pattern: str) -> List[Path]:
    city_files = sorted(raw_data_dir.glob(pattern))
    if not city_files:
        raise ValueError(f"No raw graph files found in {raw_data_dir} with pattern {pattern}")
    return city_files


def _job_label(interface_name: str, city_graph_path: str | Path, radius: float, slope: float | None = None) -> str:
    city_stem = Path(city_graph_path).stem
    base = f"{city_stem} | {interface_name} | radius={radius:g}"
    if slope is not None:
        base += f" | window_drop_slope={slope:g}"
    return base


def _run_one_interface_job(job: Dict[str, Any]) -> Dict[str, Any]:
    config_dir = Path(job["config_dir"])
    interface_cfg = job["interface_cfg"]
    city_graph_path = Path(job["city_graph_path"])
    sequence_root_dir = Path(job["sequence_root_dir"])
    radius = job["radius"]
    window_drop_slope = job.get("window_drop_slope", None)

    interface_name = interface_cfg["name"]
    print(f"[START] {_job_label(interface_name, city_graph_path, radius, window_drop_slope)}", flush=True)

    module_file = _resolve_path(config_dir, interface_cfg["module_file"])
    function_name = interface_cfg["function_name"]
    params = dict(interface_cfg.get("params", {}))
    params["radius"] = radius
    if window_drop_slope is not None and "window_drop_slope" in params:
        params["window_drop_slope"] = window_drop_slope

    interface_output_root = sequence_root_dir / interface_name
    interface_output_root.mkdir(parents=True, exist_ok=True)

    module = _load_module_from_file(module_file)
    fn = getattr(module, function_name)

    result = fn(
        graph_csv_path=str(city_graph_path),
        output_root=str(interface_output_root),
        **params,
    )

    return {
        "interface_name": interface_name,
        "city_graph_path": str(city_graph_path),
        "radius": radius,
        "window_drop_slope": window_drop_slope,
        "result": result,
    }


def _plot_one_attack_csv(
    attack_csv_path: Path,
    figure_output_path: Path,
    plot_module_file: Path,
    attack_utils_module_file: Path,
    graph_dir: Path,
) -> None:
    attack_utils_mod = _load_module_from_file(attack_utils_module_file)
    plot_mod = _load_module_from_file(plot_module_file)

    parse_attack_metadata = getattr(attack_utils_mod, "parse_attack_metadata")
    load_attack_sequence = getattr(attack_utils_mod, "load_attack_sequence")
    load_city_graph = getattr(attack_utils_mod, "load_city_graph")
    simulate_attack = getattr(attack_utils_mod, "simulate_attack")
    plot_curve = getattr(plot_mod, "plot_curve")

    meta = parse_attack_metadata(attack_csv_path, graph_dir=graph_dir)
    graph = load_city_graph(meta.graph_csv)
    attack_sequence = load_attack_sequence(attack_csv_path)
    metrics, _, _ = simulate_attack(graph, attack_sequence, meta.radius)

    subtitle = f"{meta.city} {meta.strategy or 'attack'} | radius = {meta.radius:g}"
    plot_curve(
        metrics=metrics,
        output_path=figure_output_path,
        subtitle=subtitle,
    )


def run_parallel_batch_from_json(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    config_dir = config_path.parent.resolve()

    raw_data_dir = _resolve_path(config_dir, cfg["raw_data_dir"])
    raw_data_pattern = cfg.get("raw_data_pattern", "*_Edgelist.csv")
    sequence_root_dir = _resolve_path(config_dir, cfg["sequence_root_dir"])
    figure_root_dir = _resolve_path(config_dir, cfg["figure_root_dir"])
    max_workers = int(cfg.get("max_workers", os.cpu_count() or 1))

    plot_module_file = _resolve_path(config_dir, cfg.get("plot_module_file", "plot_performance_curves_with_edgeinfo.py"))
    attack_utils_module_file = _resolve_path(config_dir, cfg.get("attack_utils_module_file", "attack_utils_with_edgeinfo.py"))

    interfaces = [item for item in cfg.get("interfaces", []) if item.get("enabled", True)]
    if not interfaces:
        raise ValueError("No enabled interfaces found in config.")

    radius_pool = cfg.get("radius_pool", [])
    normalized_radius_pool = [float(x) for x in radius_pool] if radius_pool else []

    window_drop_slope_pool = cfg.get("window_drop_slope_pool", [])
    normalized_window_drop_slope_pool = [float(x) for x in window_drop_slope_pool] if window_drop_slope_pool else []

    if normalized_window_drop_slope_pool and not normalized_radius_pool:
        raise ValueError("window_drop_slope_pool is provided, but radius_pool is empty.")

    if normalized_window_drop_slope_pool and len(normalized_window_drop_slope_pool) != len(normalized_radius_pool):
        raise ValueError(
            "window_drop_slope_pool must have the same length as radius_pool. "
            f"Got {len(normalized_window_drop_slope_pool)} vs {len(normalized_radius_pool)}."
        )

    city_graphs = _discover_city_graphs(raw_data_dir, raw_data_pattern)

    sequence_root_dir.mkdir(parents=True, exist_ok=True)
    figure_root_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[Dict[str, Any]] = []
    for interface_cfg in interfaces:
        params = interface_cfg.get("params", {})
        interface_default_radius = float(params.get("radius", 0))
        interface_default_window_drop_slope = params.get("window_drop_slope", None)
        interface_supports_slope = "window_drop_slope" in params

        if normalized_radius_pool:
            paired_radius_and_slope: List[tuple[float, float | None]] = []
            for idx, radius in enumerate(normalized_radius_pool):
                if interface_supports_slope and normalized_window_drop_slope_pool:
                    slope = normalized_window_drop_slope_pool[idx]
                else:
                    slope = interface_default_window_drop_slope
                paired_radius_and_slope.append((radius, slope))
        else:
            paired_radius_and_slope = [(interface_default_radius, interface_default_window_drop_slope)]

        for city_graph_path in city_graphs:
            for radius, window_drop_slope in paired_radius_and_slope:
                jobs.append(
                    {
                        "config_dir": str(config_dir),
                        "interface_cfg": interface_cfg,
                        "city_graph_path": str(city_graph_path),
                        "sequence_root_dir": str(sequence_root_dir),
                        "radius": radius,
                        "window_drop_slope": window_drop_slope,
                    }
                )

    total_jobs = len(jobs)
    print(f"[INFO] raw_data_dir={raw_data_dir}")
    print(
        f"[INFO] cities={len(city_graphs)}, interfaces={len(interfaces)}, "
        f"radius_count={len(normalized_radius_pool) if normalized_radius_pool else 1}, "
        f"window_drop_slope_pool_count={len(normalized_window_drop_slope_pool) if normalized_window_drop_slope_pool else 0}"
    )
    print(f"[INFO] total_jobs={total_jobs}, max_workers={max_workers}")
    print("[INFO] Job list:")
    for idx, job in enumerate(jobs, start=1):
        print(f"  [{idx:03d}/{total_jobs:03d}] {_job_label(job['interface_cfg']['name'], job['city_graph_path'], job['radius'], job.get('window_drop_slope'))}")

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(_run_one_interface_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            interface_name = job["interface_cfg"]["name"]
            city_graph_path = job["city_graph_path"]
            radius = job["radius"]
            window_drop_slope = job.get("window_drop_slope", None)
            label = _job_label(interface_name, city_graph_path, radius, window_drop_slope)

            try:
                payload = future.result()
                results.append(payload)
                completed_count += 1
                print(f"[DONE {completed_count}/{total_jobs}] {label}")
            except Exception as exc:
                failures.append(
                    {
                        "interface_name": interface_name,
                        "city_graph_path": city_graph_path,
                        "radius": radius,
                        "window_drop_slope": window_drop_slope,
                        "error": repr(exc),
                    }
                )
                completed_count += 1
                print(f"[FAIL {completed_count}/{total_jobs}] {label} | {exc}")

    plot_records: List[Dict[str, str]] = []
    plot_total = len(results)
    print(f"[INFO] plotting {plot_total} figure(s)...")
    for idx, item in enumerate(results, start=1):
        interface_name = item["interface_name"]
        attack_csv_path = Path(item["result"]["attack_sequence_csv"])
        figure_output_dir = figure_root_dir / interface_name
        figure_output_dir.mkdir(parents=True, exist_ok=True)
        figure_output_path = figure_output_dir / f"{attack_csv_path.stem}.png"

        print(f"[PLOT {idx}/{plot_total}] {attack_csv_path.stem} -> {figure_output_path.name}")
        _plot_one_attack_csv(
            attack_csv_path=attack_csv_path,
            figure_output_path=figure_output_path,
            plot_module_file=plot_module_file,
            attack_utils_module_file=attack_utils_module_file,
            graph_dir=raw_data_dir,
        )
        plot_records.append(
            {
                "interface_name": interface_name,
                "radius": str(item["radius"]),
                "window_drop_slope": None if item.get("window_drop_slope") is None else str(item["window_drop_slope"]),
                "attack_sequence_csv": str(attack_csv_path),
                "figure_path": str(figure_output_path),
            }
        )

    manifest = {
        "config_path": str(config_path),
        "raw_data_dir": str(raw_data_dir),
        "sequence_root_dir": str(sequence_root_dir),
        "figure_root_dir": str(figure_root_dir),
        "max_workers": max_workers,
        "radius_pool": normalized_radius_pool,
        "window_drop_slope_pool": normalized_window_drop_slope_pool,
        "job_count": total_jobs,
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
        "plots": plot_records,
    }

    manifest_path = sequence_root_dir / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[MANIFEST] {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-city CPU-parallel repair jobs from a JSON config.")
    parser.add_argument(
        "config",
        nargs="?",
        default="parallel_repair_config_radius_pool_progress.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    run_parallel_batch_from_json(args.config)


if __name__ == "__main__":
    main()
