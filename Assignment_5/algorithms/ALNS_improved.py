from collections import OrderedDict
from pathlib import Path
import sys
import random
import math
import time
import json
import re

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
IO_DIR = ASSIGNMENT_DIR / "io"
CORE_DIR = ASSIGNMENT_DIR / "core"
NEW_OPS_DIR = ASSIGNMENT_DIR / "new_operators"
for p in (IO_DIR, CORE_DIR, NEW_OPS_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from read_file import read_instance
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

import op1_reinsert as op1
import op2_destroy_repair as op2
import op3_or_opt as op3
import op4_drone_retiming as op4
import op8_related_destroy as op8
import op9_escape_related_large as op9

TEST_FILES_DIR = ASSIGNMENT_DIR.parent / "Test_files"
file_name = "F_100.txt"


def clone_solution(solution):
    return [solution[0][:], solution[1][:], solution[2][:]]


def load_instance(instance_path=None):
    if instance_path is None:
        instance_path = TEST_FILES_DIR / file_name
    return read_instance(str(instance_path))


def unpack_instance(instance_data):
    return {
        "T": instance_data["truck_times"],
        "D": instance_data["drone_times"],
        "flight_limit": instance_data["flight_limit"],
        "n_customers": instance_data["n_customers"],
        "depot": instance_data.get("depot_index", 0),
    }


def configure_operator_context(instance_data):
    T = instance_data["truck_times"]
    D = instance_data["drone_times"]
    fr = instance_data["flight_limit"]
    depot = instance_data.get("depot_index", 0)
    for op in (op1, op2, op3, op4, op8, op9):
        if hasattr(op, "set_operator_context"):
            op.set_operator_context(T, D, fr, depot)


def configure_operator_search_progress(progress):
    for op in (op1, op2, op3, op4, op8, op9):
        if hasattr(op, "set_search_progress"):
            op.set_search_progress(progress)


def build_evaluator(instance_data):
    ctx = unpack_instance(instance_data)
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = ctx["T"]
    calc.drone_times = ctx["D"]
    calc.flight_range = ctx["flight_limit"]
    calc.depot_index = ctx["depot"]

    checker = SolutionFeasibility(
        n_nodes=ctx["n_customers"] + 1,
        n_drones=2,
        depot_index=ctx["depot"],
        drone_times=ctx["D"],
        flight_range=ctx["flight_limit"],
    )
    return ctx, calc, checker


def to_parts_solution(solution):
    truck, drone1, drone2 = solution
    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]
    launch_indices_1 = [launch_idx + 1 for _, launch_idx, _ in drone1]
    landing_indices_1 = [land_idx + 1 for _, _, land_idx in drone1]
    launch_indices_2 = [launch_idx + 1 for _, launch_idx, _ in drone2]
    landing_indices_2 = [land_idx + 1 for _, _, land_idx in drone2]
    return {
        "part1": truck,
        "part2": drone_serving_1 + [-1] + drone_serving_2,
        "part3": launch_indices_1 + [-1] + launch_indices_2,
        "part4": landing_indices_1 + [-1] + landing_indices_2,
    }


def evaluate_solution(solution, calc, checker):
    parts_solution = to_parts_solution(solution)
    if not checker.is_solution_feasible(parts_solution):
        return False, float("inf")

    total_time, _, _, calc_feasible = calc.calculate_total_waiting_time(parts_solution)
    if not calc_feasible:
        return False, float("inf")
    return True, total_time


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def fast_precheck_solution(solution, ctx):
    truck, drone1, drone2 = solution
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    n_customers = ctx["n_customers"]
    depot = ctx["depot"]

    if len(truck) < 2 or truck[0] != depot or truck[-1] != depot:
        return False

    truck_customers = [node for node in truck if node != depot]
    if len(truck_customers) != len(set(truck_customers)):
        return False

    drone_nodes = []
    for route in (drone1, drone2):
        prev_land = 0
        used_launch = set()
        used_land = set()
        for node, launch_idx, land_idx in route:
            if node == depot:
                return False
            if not (0 <= launch_idx < land_idx < len(truck)):
                return False
            if launch_idx in used_launch or land_idx in used_land:
                return False
            if launch_idx < prev_land:
                return False

            launch_node = truck[launch_idx]
            land_node = truck[land_idx]
            if D[launch_node][node] + D[node][land_node] > flight_limit:
                return False

            used_launch.add(launch_idx)
            used_land.add(land_idx)
            prev_land = land_idx
            drone_nodes.append(node)

    if len(drone_nodes) != len(set(drone_nodes)):
        return False

    all_served = truck_customers + drone_nodes
    if len(all_served) != n_customers:
        return False
    if len(set(all_served)) != n_customers:
        return False
    return True


def format_solution_pipe(solution):
    parts = to_parts_solution(solution)
    return (
        f"{','.join(str(x) for x in parts['part1'])} | "
        f"{','.join(str(x) for x in parts['part2'])} | "
        f"{','.join(str(x) for x in parts['part3'])} | "
        f"{','.join(str(x) for x in parts['part4'])}"
    )


def _snapshot_record(iter_idx, phase, op_name, incumbent_cost, best_cost, solution):
    return {
        "iter": int(iter_idx),
        "phase": phase,
        "operator": op_name,
        "incumbent_cost": float(incumbent_cost),
        "best_cost": float(best_cost),
        "truck": solution[0][:],
        "drone1": [list(t) for t in solution[1]],
        "drone2": [list(t) for t in solution[2]],
    }


def _plot_operator_deltas(delta_points_by_op, op_names, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for op_name in op_names:
        points = delta_points_by_op.get(op_name, [])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(xs, ys, s=10, alpha=0.6, color="tab:blue")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Operator {op_name} Delta Values ({instance_label}, {run_label})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Delta Objective (new - incumbent)")
        ax.grid(alpha=0.25)
        plt.tight_layout()

        out_file = out_dir / f"delta_scatter_{safe_instance}_{safe_run}_{op_name}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        saved.append(str(out_file))

    return saved


def _plot_operator_weights(weight_history, op_names, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not weight_history:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = list(range(1, len(weight_history) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    for op_name in op_names:
        ys = [float(w.get(op_name, 0.0)) for _, w in weight_history]
        ax.plot(segments, ys, marker="o", linewidth=1.5, markersize=3, label=op_name)

    ax.set_title(f"Operator Weights by Segment ({instance_label}, {run_label})")
    ax.set_xlabel("Segment Number")
    ax.set_ylabel("Operator Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()

    out_file = out_dir / f"operator_weights_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_temperature_history(temperature_history, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not temperature_history:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [int(p[0]) for p in temperature_history]
    ys = [float(p[1]) for p in temperature_history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, linewidth=1.6, color="tab:red")
    ax.set_title(f"Temperature Across Iterations ({instance_label}, {run_label})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Temperature")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out_file = out_dir / f"temperature_plot_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _plot_acceptance_probability(acceptance_points, output_dir, instance_label, run_label):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    if not acceptance_points:
        return ""

    safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_label))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = [int(p[0]) for p in acceptance_points]
    ys = [float(p[1]) for p in acceptance_points]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(xs, ys, s=10, alpha=0.55, color="tab:purple")
    ax.set_title(f"Acceptance Probability for Worsening Moves ({instance_label}, {run_label})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Probability of Acceptance")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out_file = out_dir / f"acceptance_probability_{safe_instance}_{safe_run}.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return str(out_file)


def _init_stats(op_names):
    return {
        op_name: {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "uphill_accepted": 0,
            "uphill_rejected": 0,
            "worse_feasible": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
            "uphill_accepted_delta_sum": 0.0,
            "p_accept_sum": 0.0,
            "p_accept_count": 0,
        }
        for op_name in op_names
    }


def _normalize_weight_dict(weights, min_weight=0.03):
    fixed = {k: max(min_weight, float(v)) for k, v in weights.items()}
    s = sum(fixed.values())
    return {k: v / s for k, v in fixed.items()}


def _normalize_weight_dict_with_caps(
    weights,
    min_weight=0.03,
    lower_caps=None,
    upper_caps=None,
):
    lower_caps = lower_caps or {}
    upper_caps = upper_caps or {}
    keys = list(weights.keys())
    if not keys:
        return {}

    base = {k: max(min_weight, float(weights[k])) for k in keys}
    lower = {k: max(min_weight, float(lower_caps.get(k, min_weight))) for k in keys}
    upper = {k: float(upper_caps.get(k, 1.0)) for k in keys}

    for k in keys:
        if upper[k] < lower[k]:
            upper[k] = lower[k]

    x = {k: min(max(base[k], lower[k]), upper[k]) for k in keys}

    # Bounded re-scaling so sum(x)=1 while preserving per-operator bounds.
    for _ in range(24):
        s = sum(x.values())
        if abs(s - 1.0) <= 1e-12:
            break

        if s > 1.0:
            free = [k for k in keys if x[k] > lower[k] + 1e-12]
            if not free:
                break
            excess = s - 1.0
            room = sum(x[k] - lower[k] for k in free)
            if room <= 1e-12:
                break
            for k in free:
                take = excess * ((x[k] - lower[k]) / room)
                x[k] = max(lower[k], x[k] - take)
        else:
            free = [k for k in keys if x[k] < upper[k] - 1e-12]
            if not free:
                break
            deficit = 1.0 - s
            room = sum(upper[k] - x[k] for k in free)
            if room <= 1e-12:
                break
            for k in free:
                add = deficit * ((upper[k] - x[k]) / room)
                x[k] = min(upper[k], x[k] + add)

    # Final normalize for numerical cleanliness.
    total = sum(x.values())
    if total <= 0:
        return _normalize_weight_dict(weights, min_weight=min_weight)
    return {k: x[k] / total for k in keys}


def _phase_weight_summary(weight_history, op_names, iterations, fallback_weights):
    buckets = {
        "early": {k: [] for k in op_names},
        "mid": {k: [] for k in op_names},
        "late": {k: [] for k in op_names},
    }
    for iter_idx, w in weight_history:
        progress = (iter_idx / iterations) if iterations > 0 else 1.0
        if progress <= (1.0 / 3.0):
            phase = "early"
        elif progress <= (2.0 / 3.0):
            phase = "mid"
        else:
            phase = "late"
        for k in op_names:
            buckets[phase][k].append(float(w.get(k, 0.0)))

    summary = {}
    for phase in ("early", "mid", "late"):
        summary[phase] = {}
        for k in op_names:
            values = buckets[phase][k]
            if values:
                summary[phase][k] = sum(values) / len(values)
            else:
                summary[phase][k] = float(fallback_weights.get(k, 0.0))
    return summary


def roulette_pick(weights, op_names):
    r = random.random()
    acc = 0.0
    for op_name in op_names:
        acc += weights[op_name]
        if r <= acc:
            return op_name
    return op_names[-1]


def apply_main_operator(solution, op_name):
    if op_name == "op1":
        return op1.operator(solution)
    if op_name == "op2":
        return op2.operator(solution)
    if op_name == "op3":
        return op3.operator(solution)
    if op_name == "op8":
        return op8.operator(solution)
    return op4.operator(solution)


def _truck_removal_gain(truck, idx, truck_times):
    if idx <= 0 or idx >= len(truck) - 1:
        return -1e18
    a = truck[idx - 1]
    b = truck[idx]
    c = truck[idx + 1]
    return truck_times[a][b] + truck_times[b][c] - truck_times[a][c]


def _best_truck_insert(node, truck, truck_times):
    best_delta = None
    best_idx = None
    for ins_idx in range(1, len(truck)):
        a = truck[ins_idx - 1]
        b = truck[ins_idx]
        delta = truck_times[a][node] + truck_times[node][b] - truck_times[a][b]
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = ins_idx
    return best_delta, best_idx


def _remove_conflicting_and_shift_after_removal(route, removed_idx, removed_node):
    kept = []
    orphan_customers = []
    for cust, launch_idx, land_idx in route:
        if cust == removed_node or launch_idx == removed_idx or land_idx == removed_idx:
            orphan_customers.append(cust)
            continue
        if launch_idx > removed_idx:
            launch_idx -= 1
        if land_idx > removed_idx:
            land_idx -= 1
        kept.append((cust, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_customers


def _remove_conflicts_and_shift_after_insert(route, insert_idx):
    kept = []
    orphan_customers = []
    for cust, launch_idx, land_idx in route:
        if launch_idx == insert_idx or land_idx == insert_idx:
            orphan_customers.append(cust)
            continue
        if launch_idx >= insert_idx:
            launch_idx += 1
        if land_idx >= insert_idx:
            land_idx += 1
        kept.append((cust, launch_idx, land_idx))
    kept.sort(key=lambda x: (x[1], x[2], x[0]))
    return kept, orphan_customers


def _aggressive_escape_destroy_repair(solution, ctx):
    truck_times = ctx["T"]
    truck, drone1, drone2 = clone_solution(solution)

    if len(truck) <= 7:
        return solution

    scored = []
    for idx in range(1, len(truck) - 1):
        gain = _truck_removal_gain(truck, idx, truck_times)
        scored.append((gain, idx))
    if not scored:
        return solution

    scored.sort(key=lambda x: x[0], reverse=True)
    top10 = scored[: min(10, len(scored))]
    if len(top10) <= 5:
        picked = [idx for _, idx in top10]
    else:
        picked = [idx for _, idx in random.sample(top10, 5)]
    picked = sorted(set(picked), reverse=True)

    pending = []
    for rem_idx in picked:
        rem_node = truck[rem_idx]
        pending.append(rem_node)
        truck = truck[:rem_idx] + truck[rem_idx + 1 :]
        drone1, orphan1 = _remove_conflicting_and_shift_after_removal(drone1, rem_idx, rem_node)
        drone2, orphan2 = _remove_conflicting_and_shift_after_removal(drone2, rem_idx, rem_node)
        pending.extend(orphan1)
        pending.extend(orphan2)

    pending_unique = []
    seen = set()
    for node in pending:
        if node in seen:
            continue
        seen.add(node)
        pending_unique.append(node)

    for node in pending_unique:
        _, ins_idx = _best_truck_insert(node, truck, truck_times)
        if ins_idx is None:
            continue
        drone1, orphan1 = _remove_conflicts_and_shift_after_insert(drone1, ins_idx)
        drone2, orphan2 = _remove_conflicts_and_shift_after_insert(drone2, ins_idx)
        truck = truck[:ins_idx] + [node] + truck[ins_idx:]
        for orphan in orphan1 + orphan2:
            if orphan not in seen:
                seen.add(orphan)
                pending_unique.append(orphan)

    return [truck, drone1, drone2]


def _escape_with_related_large(
    incumbent,
    incumbent_cost,
    best_cost,
    ctx,
    cached_evaluate,
):
    current = incumbent
    current_cost = incumbent_cost
    improved_best = False
    feasible_steps = 0

    for _ in range(6):
        cand = op9.operator(current)
        if cand is current or cand == current:
            continue
        if not fast_precheck_solution(cand, ctx):
            continue
        feasible, cand_cost = cached_evaluate(cand)
        if not feasible:
            continue

        current = cand
        current_cost = cand_cost
        feasible_steps += 1
        if current_cost < best_cost:
            improved_best = True

    # Fallback: run classic destroy/repair to keep escape robust when op9 misses.
    if feasible_steps == 0:
        for _ in range(8):
            cand = op2.operator(current)
            if cand is current or cand == current:
                continue
            if not fast_precheck_solution(cand, ctx):
                continue
            feasible, cand_cost = cached_evaluate(cand)
            if not feasible:
                continue
            current = cand
            current_cost = cand_cost
            feasible_steps += 1
            if current_cost < best_cost:
                improved_best = True

    if feasible_steps == 0:
        fallback = _aggressive_escape_destroy_repair(current, ctx)
        if fallback != current and fast_precheck_solution(fallback, ctx):
            feasible, fb_cost = cached_evaluate(fallback)
            if feasible:
                current = fallback
                current_cost = fb_cost
                feasible_steps += 1
                if current_cost < best_cost:
                    improved_best = True

    return current, current_cost, improved_best, feasible_steps


def alns_improved(
    initial_solution,
    instance_data=None,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    cache_limit=200000,
    reaction_factor=0.08,
    segment_length=200,
    escape_stall_limit=650,
    ctx=None,
    calc=None,
    checker=None,
    shared_eval_cache=None,
    snapshot_on_accepted=False,
    snapshot_accept_stride=1,
    snapshot_every_iteration=False,
    snapshot_iteration_stride=25,
    reward_improve_threshold=50.0,
    uphill_reward_cap=30.0,
    sigma_global_best=8.0,
    sigma_incumbent_improve=4.0,
    sigma_small_improve=1.0,
    sigma_uphill_accepted=0.6,
    warmup_delta_trim_quantile=0.9,
    collect_delta_points=False,
    collect_temperature_history=False,
    collect_acceptance_points=False,
):
    if instance_data is None:
        instance_data = load_instance()
    if ctx is None or calc is None or checker is None:
        ctx, calc, checker = build_evaluator(instance_data)
        configure_operator_context(instance_data)

    op_names = ["op1", "op2", "op3", "op8"]

    eval_cache = shared_eval_cache if shared_eval_cache is not None else OrderedDict()

    def cached_evaluate(sol):
        key = solution_key(sol)
        cached = eval_cache.get(key)
        if cached is not None:
            eval_cache.move_to_end(key)
            return cached

        result = evaluate_solution(sol, calc, checker)
        eval_cache[key] = result
        if cache_limit is not None and cache_limit > 0 and len(eval_cache) > cache_limit:
            eval_cache.popitem(last=False)
        return result

    stats = _init_stats(op_names)

    incumbent = clone_solution(initial_solution)
    incumbent_feasible, incumbent_cost = cached_evaluate(incumbent)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = clone_solution(incumbent)
    best_cost = incumbent_cost
    best_found_iteration = 0

    total_steps = max(1, warmup_iterations + iterations)
    accepted_snapshots = []
    periodic_snapshots = []
    snapshot_accept_stride = max(1, int(snapshot_accept_stride))
    snapshot_iteration_stride = max(1, int(snapshot_iteration_stride))
    accepted_counter = 0

    deltas = []
    delta_points = {k: [] for k in op_names}
    acceptance_points = []
    for w in range(warmup_iterations):
        configure_operator_search_progress(w / total_steps)
        op_name = random.choice(op_names)
        stats[op_name]["used"] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue

        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue

        stats[op_name]["feasible"] += 1
        delta_e = cand_cost - incumbent_cost
        if collect_delta_points:
            delta_points[op_name].append((w + 1, float(delta_e)))
        if snapshot_every_iteration and ((w + 1) % snapshot_iteration_stride == 0):
            periodic_snapshots.append(
                _snapshot_record(
                    w + 1,
                    "warmup",
                    op_name,
                    cand_cost,
                    best_cost,
                    candidate,
                )
            )
        if delta_e >= 0:
            deltas.append(delta_e)
            if collect_acceptance_points:
                acceptance_points.append((w + 1, 0.8))

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e
            if snapshot_on_accepted:
                accepted_counter += 1
                if accepted_counter % snapshot_accept_stride == 0:
                    accepted_snapshots.append(
                        _snapshot_record(
                            w + 1,
                            "warmup",
                            op_name,
                            incumbent_cost,
                            best_cost,
                            incumbent,
                        )
                    )
            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                best_found_iteration = w + 1
        else:
            stats[op_name]["worse_feasible"] += 1
            if random.random() < 0.8:
                incumbent = candidate
                incumbent_cost = cand_cost
                stats[op_name]["accepted"] += 1
                stats[op_name]["uphill_accepted"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["uphill_accepted_delta_sum"] += delta_e
                if snapshot_on_accepted:
                    accepted_counter += 1
                    if accepted_counter % snapshot_accept_stride == 0:
                        accepted_snapshots.append(
                            _snapshot_record(
                                w + 1,
                                "warmup",
                                op_name,
                                incumbent_cost,
                                best_cost,
                                incumbent,
                            )
                        )
            else:
                stats[op_name]["uphill_rejected"] += 1

    if deltas:
        sorted_deltas = sorted(deltas)
        q = max(0.0, min(1.0, float(warmup_delta_trim_quantile)))
        cut_idx = int(len(sorted_deltas) * q)
        cut_idx = max(1, min(len(sorted_deltas), cut_idx))
        trimmed = sorted_deltas[:cut_idx]
        delta_avg = sum(trimmed) / len(trimmed)
    else:
        delta_avg = 1.0
    t0 = -delta_avg / math.log(0.8)
    t0_used_fallback = False
    if t0 <= 0:
        t0 = 1.0
        t0_used_fallback = True

    alpha = (final_temperature / t0) ** (1.0 / iterations) if iterations > 0 else 1.0
    temperature = t0

    weights = {k: 1.0 / len(op_names) for k in op_names}
    segment_scores = {k: 0.0 for k in op_names}
    segment_uses = {k: 0 for k in op_names}
    segment_feasible_uses = {k: 0 for k in op_names}
    weight_history = []
    temperature_history = []
    if collect_temperature_history:
        temperature_history.append((warmup_iterations, float(temperature)))

    escape_calls = 0
    escape_feasible_steps = 0
    no_best_improve_steps = 0

    for it in range(iterations):
        if collect_temperature_history:
            temperature_history.append((warmup_iterations + it + 1, float(temperature)))
        configure_operator_search_progress((warmup_iterations + it) / total_steps)
        op_name = roulette_pick(weights, op_names)
        stats[op_name]["used"] += 1
        segment_uses[op_name] += 1

        candidate = apply_main_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            temperature = alpha * temperature
            continue
        if not fast_precheck_solution(candidate, ctx):
            temperature = alpha * temperature
            continue

        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            temperature = alpha * temperature
            continue

        stats[op_name]["feasible"] += 1
        segment_feasible_uses[op_name] += 1

        delta_e = cand_cost - incumbent_cost
        if collect_delta_points:
            delta_points[op_name].append((warmup_iterations + it + 1, float(delta_e)))
        if snapshot_every_iteration and ((warmup_iterations + it + 1) % snapshot_iteration_stride == 0):
            periodic_snapshots.append(
                _snapshot_record(
                    warmup_iterations + it + 1,
                    "main",
                    op_name,
                    cand_cost,
                    best_cost,
                    candidate,
                )
            )
        accepted = False
        improved_best = False

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
            accepted = True
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e

            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                improved_best = True
                best_found_iteration = warmup_iterations + it + 1
        else:
            stats[op_name]["worse_feasible"] += 1
            p_accept = math.exp(-delta_e / temperature) if temperature > 0 else 0.0
            stats[op_name]["p_accept_sum"] += p_accept
            stats[op_name]["p_accept_count"] += 1
            if collect_acceptance_points:
                acceptance_points.append((warmup_iterations + it + 1, float(p_accept)))
            if random.random() < p_accept:
                incumbent = candidate
                incumbent_cost = cand_cost
                accepted = True
                stats[op_name]["accepted"] += 1
                stats[op_name]["uphill_accepted"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["uphill_accepted_delta_sum"] += delta_e
            else:
                stats[op_name]["uphill_rejected"] += 1

        if accepted and snapshot_on_accepted:
            accepted_counter += 1
            if accepted_counter % snapshot_accept_stride == 0:
                accepted_snapshots.append(
                    _snapshot_record(
                        warmup_iterations + it + 1,
                        "main",
                        op_name,
                        incumbent_cost,
                        best_cost,
                        incumbent,
                    )
                )

        improve_mag = max(0.0, -delta_e)
        improve_threshold = max(1.0, float(reward_improve_threshold))
        meaningful_improve = improve_mag >= improve_threshold
        magnitude_bonus = min(3.0, improve_mag / (2.0 * improve_threshold))
        segment_reward = 0.0

        if improved_best:
            segment_reward = float(sigma_global_best) + magnitude_bonus
            no_best_improve_steps = 0
        elif delta_e < 0:
            if meaningful_improve:
                segment_reward = float(sigma_incumbent_improve) + 0.5 * magnitude_bonus
            else:
                # Keep some credit for small-but-consistent improving moves.
                small_gain = improve_mag / improve_threshold
                segment_reward = float(sigma_small_improve) * max(0.0, min(1.0, small_gain))
            no_best_improve_steps += 1
        elif accepted:
            # Controlled credit for accepted uphill diversification moves.
            uphill_cap = max(1.0, float(uphill_reward_cap))
            if 0.0 < delta_e <= uphill_cap:
                uphill_quality = max(0.0, 1.0 - (delta_e / uphill_cap))
                temp_ratio = max(0.0, min(1.0, temperature / max(1.0, t0)))
                temp_scale = 0.5 + 0.5 * temp_ratio
                segment_reward = float(sigma_uphill_accepted) * uphill_quality * temp_scale
            no_best_improve_steps += 1
        else:
            no_best_improve_steps += 1

        if segment_reward > 0.0:
            segment_scores[op_name] += segment_reward

        if (it + 1) % segment_length == 0:
            updated = {}
            progress = (warmup_iterations + it + 1) / total_steps
            for k in op_names:
                theta = segment_uses[k]
                if theta > 0:
                    # Use total operator calls in the denominator so sparse-feasible operators
                    # are not over-rewarded by a few large successful moves.
                    effective_r = reaction_factor * (theta / (theta + 8.0))
                    updated[k] = weights[k] * (1.0 - effective_r) + effective_r * (
                        segment_scores[k] / theta
                    )
                else:
                    # Late-run scarcity safeguard: avoid killing an operator purely due no feasible hits.
                    if progress >= 0.7:
                        updated[k] = weights[k] * (1.0 - 0.35 * reaction_factor)
                    else:
                        updated[k] = weights[k] * (1.0 - reaction_factor)
            # Cap noisy destroy operators to keep adaptation stable:
            # - lower cap preserves exploration
            # - upper cap avoids late-run operator collapse into one heavy destroy move
            weights = _normalize_weight_dict_with_caps(
                updated,
                min_weight=0.03,
                lower_caps={"op2": 0.05, "op8": 0.05},
                upper_caps={"op2": 0.40, "op8": 0.20},
            )
            weight_history.append((it + 1, dict(weights)))
            segment_scores = {k: 0.0 for k in op_names}
            segment_uses = {k: 0 for k in op_names}
            segment_feasible_uses = {k: 0 for k in op_names}

        if no_best_improve_steps >= escape_stall_limit:
            escape_calls += 1
            incumbent, incumbent_cost, esc_improved_best, esc_steps = _escape_with_related_large(
                incumbent,
                incumbent_cost,
                best_cost,
                ctx,
                cached_evaluate,
            )
            escape_feasible_steps += esc_steps
            if esc_improved_best and incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                best_found_iteration = warmup_iterations + it + 1
                no_best_improve_steps = 0
            else:
                no_best_improve_steps = max(0, escape_stall_limit // 3)

        temperature = alpha * temperature

    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    phase_weights = _phase_weight_summary(weight_history, op_names, iterations, weights)

    stats["_meta"] = {
        "warmup_delta_avg": delta_avg,
        "warmup_delta_samples": len(deltas),
        "t0": t0,
        "t0_used_fallback": t0_used_fallback,
        "final_weights": dict(weights),
        "phase_weights": phase_weights,
        "weight_history": weight_history,
        "escape_calls": escape_calls,
        "escape_feasible_steps": escape_feasible_steps,
        "accepted_snapshots": accepted_snapshots,
        "periodic_snapshots": periodic_snapshots,
        "best_found_iteration": best_found_iteration,
        "delta_points": delta_points,
        "temperature_history": temperature_history,
        "acceptance_points": acceptance_points,
    }

    return best_solution, best_cost, stats


def run_statistics(
    initial_solution,
    instance_data=None,
    runs=10,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    cache_limit=200000,
    reaction_factor=0.15,
    segment_length=100,
    escape_stall_limit=650,
    verbose=True,
    return_metrics=False,
    print_solution_pipe=False,
    snapshot_on_accepted=False,
    snapshot_output_file=None,
    snapshot_accept_stride=1,
    snapshot_every_iteration=False,
    snapshot_iteration_stride=25,
    reward_improve_threshold=50.0,
    uphill_reward_cap=30.0,
    sigma_global_best=8.0,
    sigma_incumbent_improve=4.0,
    sigma_small_improve=1.0,
    sigma_uphill_accepted=0.6,
    warmup_delta_trim_quantile=0.9,
    plot_delta_scatter_best_run=True,
    delta_plot_output_dir=None,
    plot_weights_best_run=True,
    weight_plot_output_dir=None,
    plot_temperature_best_run=True,
    temperature_plot_output_dir=None,
    plot_acceptance_probability_best_run=True,
    acceptance_probability_output_dir=None,
    temperature_threshold=1.0,
):
    if instance_data is None:
        instance_data = load_instance()

    ctx, calc, checker = build_evaluator(instance_data)
    configure_operator_context(instance_data)

    op_names = ["op1", "op2", "op3", "op8"]

    init_feasible, init_cost = evaluate_solution(initial_solution, calc, checker)
    if not init_feasible:
        raise ValueError("Initial solution is not feasible; cannot compute improvement statistics.")

    run_costs = []
    run_times = []
    global_best_solution = None
    global_best_cost = float("inf")
    shared_eval_cache = OrderedDict()

    aggregate_stats = _init_stats(op_names)

    warmup_delta_avgs = []
    warmup_delta_samples = []
    t0_values = []
    t0_fallback_count = 0
    final_weights_acc = {k: 0.0 for k in op_names}
    phase_weights_acc = {
        "early": {k: 0.0 for k in op_names},
        "mid": {k: 0.0 for k in op_names},
        "late": {k: 0.0 for k in op_names},
    }
    total_escape_calls = 0
    total_escape_steps = 0
    best_found_iterations = []
    best_run_delta_points = None
    best_run_weight_history = None
    best_run_temperature_history = None
    best_run_acceptance_points = None
    best_run_index = None
    threshold_cross_iterations = []

    for run_id in range(runs):
        start = time.perf_counter()
        best_solution, best_cost, op_stats = alns_improved(
            clone_solution(initial_solution),
            instance_data=instance_data,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            final_temperature=final_temperature,
            cache_limit=cache_limit,
            reaction_factor=reaction_factor,
            segment_length=segment_length,
            escape_stall_limit=escape_stall_limit,
            ctx=ctx,
            calc=calc,
            checker=checker,
            shared_eval_cache=shared_eval_cache,
            snapshot_on_accepted=snapshot_on_accepted,
            snapshot_accept_stride=snapshot_accept_stride,
            snapshot_every_iteration=snapshot_every_iteration,
            snapshot_iteration_stride=snapshot_iteration_stride,
            reward_improve_threshold=reward_improve_threshold,
            uphill_reward_cap=uphill_reward_cap,
            sigma_global_best=sigma_global_best,
            sigma_incumbent_improve=sigma_incumbent_improve,
            sigma_small_improve=sigma_small_improve,
            sigma_uphill_accepted=sigma_uphill_accepted,
            warmup_delta_trim_quantile=warmup_delta_trim_quantile,
            collect_delta_points=plot_delta_scatter_best_run,
            collect_temperature_history=plot_temperature_best_run,
            collect_acceptance_points=plot_acceptance_probability_best_run,
        )
        elapsed = time.perf_counter() - start

        for op_name in op_names:
            for k in aggregate_stats[op_name]:
                aggregate_stats[op_name][k] += op_stats[op_name][k]

        run_meta = op_stats.get("_meta")
        if run_meta is not None:
            warmup_delta_avgs.append(run_meta.get("warmup_delta_avg", 0.0))
            warmup_delta_samples.append(run_meta.get("warmup_delta_samples", 0))
            t0_values.append(run_meta.get("t0", 0.0))
            if run_meta.get("t0_used_fallback", False):
                t0_fallback_count += 1
            fw = run_meta.get("final_weights", {})
            for k in op_names:
                final_weights_acc[k] += float(fw.get(k, 0.0))
            pw = run_meta.get("phase_weights", {})
            for phase in ("early", "mid", "late"):
                phase_map = pw.get(phase, {})
                for k in op_names:
                    phase_weights_acc[phase][k] += float(phase_map.get(k, 0.0))
            total_escape_calls += int(run_meta.get("escape_calls", 0))
            total_escape_steps += int(run_meta.get("escape_feasible_steps", 0))
            best_found_iterations.append(int(run_meta.get("best_found_iteration", 0)))
            temp_hist = run_meta.get("temperature_history", [])
            cross_iter = 0
            thr = float(temperature_threshold)
            for iter_idx, temp_val in temp_hist:
                if float(temp_val) <= thr:
                    cross_iter = int(iter_idx)
                    break
            threshold_cross_iterations.append(cross_iter)
            if (snapshot_on_accepted or snapshot_every_iteration) and snapshot_output_file and run_id == 0:
                chosen_snapshots = (
                    run_meta.get("periodic_snapshots", [])
                    if snapshot_every_iteration
                    else run_meta.get("accepted_snapshots", [])
                )
                payload = {
                    "instance": file_name,
                    "run_id": 1,
                    "warmup_iterations": warmup_iterations,
                    "iterations": iterations,
                    "initial_score": init_cost,
                    "snapshot_accept_stride": int(snapshot_accept_stride),
                    "snapshot_iteration_stride": int(snapshot_iteration_stride),
                    "snapshots": chosen_snapshots,
                }
                out_path = Path(snapshot_output_file)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)

        run_costs.append(best_cost)
        run_times.append(elapsed)

        if verbose:
            best_iter = best_found_iterations[-1] if best_found_iterations else 0
            print(
                f"Best score in run {run_id + 1}/{runs}: {best_cost} "
                f"(best found at iteration i*={best_iter})"
            )

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = clone_solution(best_solution)
            best_run_meta = op_stats.get("_meta") or {}
            best_run_delta_points = best_run_meta.get("delta_points", None)
            best_run_weight_history = best_run_meta.get("weight_history", None)
            best_run_temperature_history = best_run_meta.get("temperature_history", None)
            best_run_acceptance_points = best_run_meta.get("acceptance_points", None)
            best_run_index = run_id + 1

    avg_obj = sum(run_costs) / len(run_costs)
    best_obj = global_best_cost
    improvement_abs = init_cost - best_obj
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)

    avg_final_weights = {k: (final_weights_acc[k] / runs if runs > 0 else 0.0) for k in op_names}
    avg_phase_weights = {
        phase: {k: (phase_weights_acc[phase][k] / runs if runs > 0 else 0.0) for k in op_names}
        for phase in ("early", "mid", "late")
    }

    metrics = {
        "initial_score": init_cost,
        "average_score": avg_obj,
        "best_score": best_obj,
        "improvement_abs": improvement_abs,
        "improvement_pct": improvement_pct,
        "average_runtime": avg_runtime_per_run,
        "warmup_delta_avg_mean": (
            sum(warmup_delta_avgs) / len(warmup_delta_avgs) if warmup_delta_avgs else 0.0
        ),
        "warmup_delta_samples_mean": (
            sum(warmup_delta_samples) / len(warmup_delta_samples) if warmup_delta_samples else 0.0
        ),
        "t0_mean": (sum(t0_values) / len(t0_values) if t0_values else 0.0),
        "t0_fallback_runs": t0_fallback_count,
        "avg_final_weights": avg_final_weights,
        "avg_phase_weights": avg_phase_weights,
        "escape_calls": total_escape_calls,
        "escape_steps": total_escape_steps,
        "best_found_iterations": best_found_iterations,
        "temperature_threshold": float(temperature_threshold),
        "temperature_threshold_cross_iterations": threshold_cross_iterations,
        "temperature_threshold_cross_mean": (
            sum(x for x in threshold_cross_iterations if x > 0)
            / max(1, len([x for x in threshold_cross_iterations if x > 0]))
        ) if threshold_cross_iterations else 0.0,
    }

    delta_plot_files = []
    if plot_delta_scatter_best_run and best_run_delta_points is not None:
        plot_dir = (
            delta_plot_output_dir
            if delta_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "delta_plots")
        )
        delta_plot_files = _plot_operator_deltas(
            best_run_delta_points,
            op_names,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["delta_plot_files"] = delta_plot_files

    weight_plot_file = ""
    if plot_weights_best_run and best_run_weight_history is not None:
        plot_dir = (
            weight_plot_output_dir
            if weight_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "weight_plots")
        )
        weight_plot_file = _plot_operator_weights(
            best_run_weight_history,
            op_names,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["weight_plot_file"] = weight_plot_file

    temperature_plot_file = ""
    if plot_temperature_best_run and best_run_temperature_history is not None:
        plot_dir = (
            temperature_plot_output_dir
            if temperature_plot_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "temperature_plots")
        )
        temperature_plot_file = _plot_temperature_history(
            best_run_temperature_history,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["temperature_plot_file"] = temperature_plot_file

    acceptance_probability_plot_file = ""
    if plot_acceptance_probability_best_run and best_run_acceptance_points is not None:
        plot_dir = (
            acceptance_probability_output_dir
            if acceptance_probability_output_dir is not None
            else str(ASSIGNMENT_DIR / "outputs" / "acceptance_probability_plots")
        )
        acceptance_probability_plot_file = _plot_acceptance_probability(
            best_run_acceptance_points,
            plot_dir,
            file_name,
            f"run{best_run_index}",
        )
    metrics["acceptance_probability_plot_file"] = acceptance_probability_plot_file

    if verbose:
        print(f"Average score: {avg_obj}")
        print(f"Best score: {best_obj}")
        print(f"Average runtime: {avg_runtime_per_run:.4f} seconds")
        print(
            f"Improvement over initial solution: {improvement_abs} "
            f"({improvement_pct:.2f}% reduction)"
        )
        if t0_values:
            print(
                "SA temperature diagnostics: "
                f"warmup_delta_avg_mean={sum(warmup_delta_avgs)/len(warmup_delta_avgs):.4f}, "
                f"warmup_samples_mean={sum(warmup_delta_samples)/len(warmup_delta_samples):.1f}, "
                f"t0_mean={sum(t0_values)/len(t0_values):.4f}, "
                f"t0_min={min(t0_values):.4f}, t0_max={max(t0_values):.4f}, "
                f"t0_fallback_runs={t0_fallback_count}/{runs}"
            )
            crossed = [x for x in threshold_cross_iterations if x > 0]
            if crossed:
                print(
                    f"Temperature threshold diagnostics (T<={float(temperature_threshold):.4f}): "
                    f"cross_mean={sum(crossed)/len(crossed):.1f}, "
                    "cross_iters=" + ", ".join(str(x) for x in threshold_cross_iterations)
                )
            else:
                print(
                    f"Temperature threshold diagnostics (T<={float(temperature_threshold):.4f}): "
                    f"not crossed in {runs}/{runs} runs"
                )

        final_w_str = ", ".join([f"{k}={avg_final_weights[k]:.3f}" for k in op_names])
        early_w_str = ", ".join([f"{k}={avg_phase_weights['early'][k]:.3f}" for k in op_names])
        mid_w_str = ", ".join([f"{k}={avg_phase_weights['mid'][k]:.3f}" for k in op_names])
        late_w_str = ", ".join([f"{k}={avg_phase_weights['late'][k]:.3f}" for k in op_names])
        print(
            "ALNS adaptation diagnostics: "
            f"avg_final_weights=({final_w_str}), "
            f"escape_calls={total_escape_calls}, escape_feasible_steps={total_escape_steps}"
        )
        if best_found_iterations:
            print(
                "Best solution iterations (i* per run): "
                + ", ".join(str(x) for x in best_found_iterations)
            )
        if delta_plot_files:
            print("Delta plots saved:")
            for fp in delta_plot_files:
                print(f"  {fp}")
        if weight_plot_file:
            print("Operator weights plot saved:")
            print(f"  {weight_plot_file}")
        if temperature_plot_file:
            print("Temperature plot saved:")
            print(f"  {temperature_plot_file}")
        if acceptance_probability_plot_file:
            print("Acceptance probability plot saved:")
            print(f"  {acceptance_probability_plot_file}")
        print(
            "ALNS phase weights: "
            f"early({early_w_str}), mid({mid_w_str}), late({late_w_str})"
        )

        if print_solution_pipe and global_best_solution is not None:
            print("Solution on pipe format:")
            print(format_solution_pipe(global_best_solution))

        print("Operator contribution stats (aggregated):")
        for op_name in op_names:
            s = aggregate_stats[op_name]
            used = s["used"]
            feasible_moves = s["feasible"]
            accepted = s["accepted"]
            improved = s["improved"]
            uphill_accepted = s["uphill_accepted"]
            uphill_rejected = s["uphill_rejected"]
            worse_feasible = s["worse_feasible"]
            avg_delta = s["delta_sum"] / accepted if accepted > 0 else float("nan")
            feasible_rate = (feasible_moves / used * 100.0) if used > 0 else 0.0
            accept_rate = (accepted / feasible_moves * 100.0) if feasible_moves > 0 else 0.0
            improve_rate = (improved / accepted * 100.0) if accepted > 0 else 0.0
            uphill_rate = (uphill_accepted / accepted * 100.0) if accepted > 0 else 0.0
            uphill_of_worse_rate = (
                uphill_accepted / worse_feasible * 100.0 if worse_feasible > 0 else 0.0
            )
            avg_uphill_accepted_delta = (
                s["uphill_accepted_delta_sum"] / uphill_accepted if uphill_accepted > 0 else float("nan")
            )
            avg_improve_accepted_delta = (
                s["improve_delta_sum"] / improved if improved > 0 else float("nan")
            )
            mean_p_accept = s["p_accept_sum"] / s["p_accept_count"] if s["p_accept_count"] > 0 else float("nan")
            improve_per_1k_uses = (s["improve_delta_sum"] * 1000.0 / used) if used > 0 else 0.0

            print(
                f"  {op_name}: used={used}, feasible={feasible_moves} ({feasible_rate:.1f}%), "
                f"accepted={accepted} ({accept_rate:.1f}%), improved={improved} ({improve_rate:.1f}%), "
                f"uphill_accepted={uphill_accepted} ({uphill_rate:.1f}%), "
                f"worse_feasible={worse_feasible}, uphill_accept_of_worse={uphill_of_worse_rate:.1f}%, "
                f"uphill_rejected={uphill_rejected}, mean_p_accept={mean_p_accept:.4f}, "
                f"avg_uphill_accepted_delta={avg_uphill_accepted_delta:.4f}, "
                f"avg_improve_accepted_delta={avg_improve_accepted_delta:.4f}, "
                f"avg_accepted_delta={avg_delta:.4f}, improve_per_1000_uses={improve_per_1k_uses:.2f}"
            )

    if return_metrics:
        return global_best_solution, global_best_cost, metrics
    return global_best_solution, global_best_cost


def main():
    instance_data = load_instance()
    n_customers = instance_data["n_customers"]
    initial_solution = [[i for i in range(n_customers + 1)] + [0], [], []]

    run_statistics(
        initial_solution,
        instance_data=instance_data,
        runs=10,
        warmup_iterations=500,
        iterations=9500,
        final_temperature=0.1,
        cache_limit=200000,
        reaction_factor=0.15,
        segment_length=100,
        escape_stall_limit=650,
        verbose=True,
        return_metrics=False,
        print_solution_pipe=False,
    )


if __name__ == "__main__":
    main()
