from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Sequence, Set, Tuple
import networkx as nx


def compute_robustness_union_find(
    graph: nx.Graph,
    attack_sequence: Sequence[int],
    target_ratio: float = 0.01,
    radius: float = 0.0,
    plot_curve: bool = True,
    verbose: bool = True,
    save_outputs: bool = True,
    output_root: str | Path = "results",
    run_name: str | None = None,
) -> Dict[str, Any]:
    """
    基于 Union-Find（并查集）+ 逆序恢复，计算给定攻击序列下的性能曲线与健壮性。

    第三问（radius=0）：
        每一步仅删除攻击节点自己。

    第四问（radius>0）：
        每一步删除攻击节点以及其空间半径 radius 内的所有节点。
        半径按节点的 (x, y) 坐标做欧氏距离计算。

    参数
    ----
    graph : nx.Graph
        输入交通网络图。
    attack_sequence : Sequence[int]
        攻击序列，按攻击顺序给出。
    target_ratio : float, default=0.01
        攻击目标阈值。当当前最大连通分量 / 原始节点总数 <= target_ratio 时提前停止。
    radius : float, default=0.0
        空间攻击半径。
        - radius = 0 时，只删除攻击节点自己；
        - radius > 0 时，删除攻击节点及其半径范围内所有节点。
    plot_curve : bool, default=True
        是否绘制性能曲线。
    verbose : bool, default=True
        是否在函数内部打印测算结果。
    save_outputs : bool, default=True
        是否保存本次运行结果。
    output_root : str | Path, default="results"
        结果保存根目录。
    run_name : str | None, default=None
        本次运行结果文件夹名；若为 None，则自动用时间戳命名。

    返回
    ----
    result : dict
        {
            "fractions_removed": List[float],
            "performance_curve": List[float],
            "robustness": float,
            "largest_component_sizes": List[int],
            "attack_sequence_used": List[int],          # 实际生效的攻击中心序列
            "removed_count_per_attack": List[int],      # 每次攻击实际删掉多少节点
            "cumulative_removed_counts": List[int],     # 累计已删除节点数（与攻击步同步）
            "n_original": int,
            "stopped_early": bool,
            "stop_step": int,
            "final_ratio": float,
            "target_ratio": float,
            "radius": float,
            "output_dir": str | None,
        }
    """
    if not (0 <= target_ratio <= 1):
        raise ValueError("target_ratio must be between 0 and 1.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if graph.number_of_nodes() == 0:
        return {
            "fractions_removed": [0.0],
            "performance_curve": [0.0],
            "robustness": 0.0,
            "largest_component_sizes": [0],
            "attack_sequence_used": [],
            "removed_count_per_attack": [],
            "cumulative_removed_counts": [],
            "n_original": 0,
            "stopped_early": True,
            "stop_step": 0,
            "final_ratio": 0.0,
            "target_ratio": target_ratio,
            "radius": radius,
            "output_dir": None,
        }

    node_set: Set[int] = set(graph.nodes())

    # 仅过滤非法节点；不在这里去重，因为 radius>0 时节点可能被前一步波及删除
    attack: List[int] = [node for node in attack_sequence if node in node_set]

    n = graph.number_of_nodes()

    # 预取坐标（radius>0 时会用到）
    coords: Dict[int, Tuple[float, float]] = {}
    if radius > 0:
        missing_coord_nodes = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if "x" not in node_data or "y" not in node_data:
                missing_coord_nodes.append(node)
            else:
                coords[node] = (float(node_data["x"]), float(node_data["y"]))

        if missing_coord_nodes:
            raise ValueError(
                f"radius > 0 时需要所有节点都有坐标，但发现 {len(missing_coord_nodes)} 个节点缺少 x/y 属性。"
            )

    # ---------------------------------------------------------
    # 第一步：按顺序模拟“空间攻击”，得到每一步真正删掉的节点组
    # ---------------------------------------------------------
    alive: Set[int] = set(graph.nodes())
    attack_groups: List[List[int]] = []          # 每次攻击真正删掉的节点集合
    effective_attack_sequence: List[int] = []    # 真正生效的攻击中心
    removed_count_per_attack: List[int] = []
    cumulative_removed_counts: List[int] = []

    radius_cache: Dict[int, Set[int]] = {}
    radius_sq = radius * radius

    def get_nodes_within_radius(center: int) -> Set[int]:
        """
        返回原图中距离 center 不超过 radius 的所有节点集合。
        radius=0 时只返回 center 自己。
        """
        if radius <= 0:
            return {center}

        if center in radius_cache:
            return radius_cache[center]

        cx, cy = coords[center]
        affected: Set[int] = set()

        for node, (x, y) in coords.items():
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= radius_sq + 1e-12:
                affected.add(node)

        radius_cache[center] = affected
        return affected

    cumulative_removed = 0
    for center in attack:
        if radius <= 0:
            removed_now = [center] if center in alive else []
        else:
            affected_all = get_nodes_within_radius(center)
            removed_now = [node for node in affected_all if node in alive]

        if not removed_now:
            continue

        removed_now.sort()  # 保证结果可复现

        for node in removed_now:
            alive.remove(node)

        cumulative_removed += len(removed_now)

        effective_attack_sequence.append(center)
        attack_groups.append(removed_now)
        removed_count_per_attack.append(len(removed_now))
        cumulative_removed_counts.append(cumulative_removed)

    m = len(effective_attack_sequence)

    if m == 0:
        largest_cc = len(max(nx.connected_components(graph), key=len)) if n > 0 else 0
        performance_curve = [largest_cc / n]
        return {
            "fractions_removed": [0.0],
            "performance_curve": performance_curve,
            "robustness": 0.0,
            "largest_component_sizes": [largest_cc],
            "attack_sequence_used": [],
            "removed_count_per_attack": [],
            "cumulative_removed_counts": [],
            "n_original": n,
            "stopped_early": performance_curve[0] <= target_ratio,
            "stop_step": 0,
            "final_ratio": performance_curve[0],
            "target_ratio": target_ratio,
            "radius": radius,
            "output_dir": None,
        }

    removed_set = set(graph.nodes()) - alive

    # ---------------------------------------------------------
    # 第二步：并查集，先构造“所有攻击做完后的残图”
    # ---------------------------------------------------------
    parent: Dict[int, int] = {}
    size: Dict[int, int] = {}
    active: Set[int] = set()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> int:
        ra, rb = find(a), find(b)
        if ra == rb:
            return size[ra]
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return size[ra]

    def add_node(x: int) -> None:
        parent[x] = x
        size[x] = 1
        active.add(x)

    largest_cc = 0
    for node in graph.nodes():
        if node not in removed_set:
            add_node(node)

    for u in active:
        for v in graph.neighbors(u):
            if v in active and u < v:
                merged_size = union(u, v)
                if merged_size > largest_cc:
                    largest_cc = merged_size

    if active and largest_cc == 0:
        largest_cc = 1

    # reverse_lcc[k] = 正向执行前 k 次“空间攻击”后的最大连通分量
    reverse_lcc = [0] * (m + 1)
    reverse_lcc[m] = largest_cc

    # ---------------------------------------------------------
    # 第三步：逆序恢复，每次恢复一整组被半径攻击删掉的节点
    # ---------------------------------------------------------
    for idx in range(m - 1, -1, -1):
        group = attack_groups[idx]

        # 先把本次攻击删掉的所有节点都加回来
        for node in group:
            add_node(node)

        if largest_cc < 1:
            largest_cc = 1

        # 再统一连边（组内、组外都会自动处理）
        for node in group:
            for nbr in graph.neighbors(node):
                if nbr in active and nbr != node:
                    merged_size = union(node, nbr)
                    if merged_size > largest_cc:
                        largest_cc = merged_size

            root = find(node)
            if size[root] > largest_cc:
                largest_cc = size[root]

        reverse_lcc[idx] = largest_cc

    # ---------------------------------------------------------
    # 第四步：转回正向性能曲线
    # 横轴：累计删除节点比例
    # ---------------------------------------------------------
    full_largest_component_sizes = reverse_lcc[:]
    full_performance_curve = [s / n for s in full_largest_component_sizes]
    full_fractions_removed = [0.0] + [cnt / n for cnt in cumulative_removed_counts]

    # 找到是否提前达到目标
    stop_step = m
    stopped_early = False
    for k, ratio in enumerate(full_performance_curve):
        if ratio <= target_ratio:
            stop_step = k
            stopped_early = True
            break

    # 截断到实际停止位置
    used_largest_component_sizes = full_largest_component_sizes[: stop_step + 1]
    used_performance_curve = full_performance_curve[: stop_step + 1]
    used_fractions_removed = full_fractions_removed[: stop_step + 1]
    used_attack_sequence = effective_attack_sequence[:stop_step]
    used_removed_count_per_attack = removed_count_per_attack[:stop_step]
    used_cumulative_removed_counts = cumulative_removed_counts[:stop_step]

    # 梯形积分
    robustness = 0.0
    for i in range(len(used_performance_curve) - 1):
        dx = used_fractions_removed[i + 1] - used_fractions_removed[i]
        robustness += 0.5 * (used_performance_curve[i] + used_performance_curve[i + 1]) * dx

    final_ratio = used_performance_curve[-1]
    actual_removed_count = used_cumulative_removed_counts[-1] if used_cumulative_removed_counts else 0

    # 提前创建保存目录，供画图时直接保存
    output_dir = None
    run_dir = None
    if save_outputs:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(run_dir)

    if plot_curve:
        plt.figure(figsize=(8, 5))
        plt.plot(used_fractions_removed, used_performance_curve, linewidth=2)
        plt.xlabel("Fraction of removed nodes")
        plt.ylabel("Largest connected component / N")
        plt.title("Performance Curve")
        plt.grid(True)
        plt.tight_layout()

        if save_outputs and output_dir is not None:
            plt.savefig(Path(output_dir) / "performance_curve.png", dpi=200)

        plt.show()

    # ===== 保存结果 =====
    if save_outputs and run_dir is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 保存基本信息
        basic_info = {
            "run_name": run_dir.name,
            "timestamp": timestamp,
            "n_original": n,
            "generated_attack_sequence_length": len(attack),
            "executed_attack_count": len(used_attack_sequence),
            "removed_count": actual_removed_count,
            "final_ratio": final_ratio,
            "robustness": robustness,
            "target_ratio": target_ratio,
            "radius": radius,
            "stopped_early": stopped_early,
            "stop_step": stop_step,
        }
        with (run_dir / "basic_info.json").open("w", encoding="utf-8") as f:
            json.dump(basic_info, f, ensure_ascii=False, indent=2)

        # 2. 保存实际执行的攻击中心序列
        with (run_dir / "attack_sequence_used.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "node", "removed_count_this_step", "cumulative_removed_count"])
            cumulative = 0
            for i, node in enumerate(used_attack_sequence, start=1):
                removed_this_step = used_removed_count_per_attack[i - 1]
                cumulative += removed_this_step
                writer.writerow([i, node, removed_this_step, cumulative])

        # 3. 保存性能曲线表格
        with (run_dir / "performance_curve.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "cumulative_removed_count", "fraction_removed", "performance_ratio"])
            for i in range(len(used_performance_curve)):
                cumulative = 0 if i == 0 else used_cumulative_removed_counts[i - 1]
                writer.writerow([i, cumulative, used_fractions_removed[i], used_performance_curve[i]])

        # 4. 保存最大连通分量变化表格
        with (run_dir / "largest_component_sizes.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "cumulative_removed_count", "fraction_removed", "largest_component_size"])
            for i in range(len(used_largest_component_sizes)):
                cumulative = 0 if i == 0 else used_cumulative_removed_counts[i - 1]
                writer.writerow([i, cumulative, used_fractions_removed[i], used_largest_component_sizes[i]])

        # 5. 保存最大连通分量变化图片
        plt.figure(figsize=(8, 5))
        plt.plot(used_fractions_removed, used_largest_component_sizes, linewidth=2)
        plt.xlabel("Fraction of removed nodes")
        plt.ylabel("Largest connected component size")
        plt.title("Largest Connected Component Size")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(run_dir / "largest_component_sizes.png", dpi=200)
        plt.close()

        # 6. 维护总表 run_history.csv
        history_path = output_root / "run_history.csv"
        history_exists = history_path.exists()
        with history_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not history_exists:
                writer.writerow([
                    "run_name",
                    "timestamp",
                    "n_original",
                    "generated_attack_sequence_length",
                    "executed_attack_count",
                    "removed_count",
                    "final_ratio",
                    "robustness",
                    "target_ratio",
                    "radius",
                    "stopped_early",
                    "stop_step",
                ])
            writer.writerow([
                run_dir.name,
                timestamp,
                n,
                len(attack),
                len(used_attack_sequence),
                actual_removed_count,
                final_ratio,
                robustness,
                target_ratio,
                radius,
                stopped_early,
                stop_step,
            ])

    if verbose:
        print("\n=== 攻击结果 ===")
        print("原始节点数:", n)
        print("生成攻击序列长度:", len(attack))
        print("实际执行攻击次数:", len(used_attack_sequence))
        print("实际删除节点数:", actual_removed_count)
        print("空间攻击半径:", radius)
        print("是否提前达到目标:", stopped_early)
        print("停止时已攻击次数:", stop_step)
        print("最后状态比值:", final_ratio)
        print("健壮性:", robustness)

        if stopped_early:
            print(
                f"\n已达到攻击目标：当前最大连通分量 / 初始节点总数 = "
                f"{final_ratio:.6f} <= {target_ratio:.6f}"
            )
            print("因此后续攻击不再继续。")
        else:
            print(
                f"\n跑完整个攻击序列后，仍未达到攻击目标："
                f"最后一次攻击后最大连通分量 / 初始节点总数 = {final_ratio:.6f}"
            )

        if save_outputs and output_dir is not None:
            print("结果已保存到目录:", output_dir)

    return {
        "fractions_removed": used_fractions_removed,
        "performance_curve": used_performance_curve,
        "robustness": robustness,
        "largest_component_sizes": used_largest_component_sizes,
        "attack_sequence_used": used_attack_sequence,
        "removed_count_per_attack": used_removed_count_per_attack,
        "cumulative_removed_counts": used_cumulative_removed_counts,
        "n_original": n,
        "stopped_early": stopped_early,
        "stop_step": stop_step,
        "final_ratio": final_ratio,
        "target_ratio": target_ratio,
        "radius": radius,
        "output_dir": output_dir,
    }
