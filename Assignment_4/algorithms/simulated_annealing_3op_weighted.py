from collections import OrderedDict
from pathlib import Path
import sys
import random
import math
import time

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
IO_DIR = ASSIGNMENT_DIR / "io"
OPS_DIR = ASSIGNMENT_DIR / "operators"
CORE_DIR = ASSIGNMENT_DIR / "core"
for p in (IO_DIR, OPS_DIR, CORE_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from read_file import read_instance
import op1_reinsert as op1
import op8_or_opt3 as op2
import op6_destroy as op3
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "F_100.txt"


def clone_solution(solution):
    # Top-level lists are mutable; inner drone tuples are immutable.
    return [solution[0][:], solution[1][:], solution[2][:]]


def load_instance(instance_path=None):
    if instance_path is None:
        instance_path = f"{absolute_path}{file_name}"
    return read_instance(instance_path)


def unpack_instance(instance_data):
    return {
        "T": instance_data["truck_times"],
        "D": instance_data["drone_times"],
        "flight_limit": instance_data["flight_limit"],
        "n_customers": instance_data["n_customers"],
        "depot": instance_data.get("depot_index", 0),
    }


def configure_operator_context(instance_data):
    truck_times = instance_data["truck_times"]
    drone_times = instance_data["drone_times"]
    range_limit = instance_data["flight_limit"]
    depot_idx = instance_data.get("depot_index", 0)

    op1.set_operator_context(truck_times, drone_times, range_limit, depot_idx)
    op2.set_operator_context(truck_times, drone_times, range_limit, depot_idx)
    op3.set_operator_context(truck_times, drone_times, range_limit, depot_idx)


def configure_operator_search_progress(progress):
    for op in (op1, op2, op3):
        if hasattr(op, "set_search_progress"):
            op.set_search_progress(progress)


def build_evaluator(instance_data):
    ctx = unpack_instance(instance_data)
    T = ctx["T"]
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    n_customers = ctx["n_customers"]
    depot = ctx["depot"]

    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = flight_limit
    calc.depot_index = depot

    checker = SolutionFeasibility(
        n_nodes=n_customers + 1,
        n_drones=2,
        depot_index=depot,
        drone_times=D,
        flight_range=flight_limit,
    )
    return ctx, calc, checker


def reset_operator_state():
    if hasattr(op1, "reset_operator_state"):
        op1.reset_operator_state()


def configure_reinsert_bias(op1_truck_to_drone_bias=None):
    if op1_truck_to_drone_bias is None:
        return
    if hasattr(op1, "set_truck_to_drone_bias"):
        op1.set_truck_to_drone_bias(op1_truck_to_drone_bias)


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


def _embed_2d_from_distance_matrix(distance_matrix):
    import numpy as np

    d = np.array(distance_matrix, dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    vals = np.maximum(eigvals[:2], 0.0)
    vecs = eigvecs[:, :2]
    coords = vecs * np.sqrt(vals)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    return coords


def plot_solution(solution, instance_data, title="SA Solution"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plot.")
        return

    truck, drone1, drone2 = solution
    coords = _embed_2d_from_distance_matrix(instance_data["truck_times"])
    depot = instance_data.get("depot_index", 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(coords[:, 0], coords[:, 1], s=30, c="black", alpha=0.7)
    ax.scatter([coords[depot, 0]], [coords[depot, 1]], s=120, c="red", marker="s", label="Depot")

    for i in range(len(truck) - 1):
        a = truck[i]
        b = truck[i + 1]
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            color="tab:blue",
            linewidth=1.8,
            alpha=0.9,
        )

    for node, launch_idx, land_idx in drone1:
        launch_node = truck[launch_idx]
        land_node = truck[land_idx]
        ax.plot(
            [coords[launch_node, 0], coords[node, 0], coords[land_node, 0]],
            [coords[launch_node, 1], coords[node, 1], coords[land_node, 1]],
            color="tab:orange",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )

    for node, launch_idx, land_idx in drone2:
        launch_node = truck[launch_idx]
        land_node = truck[land_idx]
        ax.plot(
            [coords[launch_node, 0], coords[node, 0], coords[land_node, 0]],
            [coords[launch_node, 1], coords[node, 1], coords[land_node, 1]],
            color="tab:green",
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
        )

    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def fast_precheck_solution(solution, ctx):
    truck, drone1, drone2 = solution
    D = ctx["D"]
    flight_limit = ctx["flight_limit"]
    n_customers = ctx["n_customers"]
    depot = ctx["depot"]

    truck_len = len(truck)
    if truck_len < 2 or truck[0] != depot or truck[-1] != depot:
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
            if not (0 <= launch_idx < land_idx < truck_len):
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


def _normalize_operator_weights(operator_weights):
    """
    Accepts either:
    - dict: {"op1": w1, "op2": w2, "op3": w3}
    - tuple/list: [w1, w2, w3]
    Returns normalized tuple (w1, w2, w3).
    """
    if operator_weights is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    if isinstance(operator_weights, dict):
        w1 = float(operator_weights.get("op1", 0.0))
        w2 = float(operator_weights.get("op2", 0.0))
        w3 = float(operator_weights.get("op3", 0.0))
    else:
        if len(operator_weights) != 3:
            raise ValueError("operator_weights must contain 3 values: [w1, w2, w3]")
        w1 = float(operator_weights[0])
        w2 = float(operator_weights[1])
        w3 = float(operator_weights[2])

    if w1 < 0 or w2 < 0 or w3 < 0:
        raise ValueError("operator weights must be non-negative")

    total = w1 + w2 + w3
    if total <= 0:
        raise ValueError("sum of operator weights must be > 0")

    return (w1 / total, w2 / total, w3 / total)


def apply_weighted_operator(solution, operator_weights=None):
    """
    Weighted operator selection for 3 operators.
    """
    if (
        isinstance(operator_weights, tuple)
        and len(operator_weights) == 3
        and abs(sum(operator_weights) - 1.0) < 1e-12
    ):
        w1, w2, w3 = operator_weights
    else:
        w1, w2, w3 = _normalize_operator_weights(operator_weights)
    r = random.random()
    if r < w1:
        return op1.operator(solution), "op1_used"
    if r < (w1 + w2):
        return op2.truck_2opt(solution), "op2_used"
    return op3.operator(solution), "op3_used"


def simulated_annealing(
    initial_solution,
    instance_data=None,
    warmup_iterations=100,
    iterations=9900,
    final_temperature=0.1,
    cache_limit=200000,
    operator_weights={"op1": 0.2, "op2": 0.5, "op3": 0.3},
    op1_truck_to_drone_bias=None,
    ctx=None,
    calc=None,
    checker=None,
    shared_eval_cache=None,
):
    """
    Same SA flow as simulated_annealing_3op.py, but with configurable operator weights.
    """
    if instance_data is None:
        instance_data = load_instance()
    if ctx is None or calc is None or checker is None:
        ctx, calc, checker = build_evaluator(instance_data)
        configure_operator_context(instance_data)

    configure_reinsert_bias(op1_truck_to_drone_bias)
    reset_operator_state()
    normalized_weights = _normalize_operator_weights(operator_weights)

    eval_cache = shared_eval_cache if shared_eval_cache is not None else OrderedDict()
    stats = {
        "op1": {
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
        },
        "op2": {
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
        },
        "op3": {
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
        },
    }

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

    incumbent = clone_solution(initial_solution)
    incumbent_feasible, incumbent_cost = cached_evaluate(incumbent)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = clone_solution(incumbent)
    best_cost = incumbent_cost
    deltas = []
    total_steps = max(1, warmup_iterations + iterations)

    for w in range(warmup_iterations):
        configure_operator_search_progress(w / total_steps)
        #Returns the new solution based on the used operator and the key of that operator
        new_solution, op_key = apply_weighted_operator(incumbent, operator_weights=normalized_weights)
        op_name = op_key.split("_")[0]
        stats[op_name]["used"] += 1
        if new_solution is incumbent or new_solution == incumbent:
            continue
        if not fast_precheck_solution(new_solution, ctx):
            continue
        feasible, new_cost = cached_evaluate(new_solution)
        if not feasible:
            continue
        stats[op_name]["feasible"] += 1

        delta_e = new_cost - incumbent_cost
        if delta_e >= 0:
            deltas.append(delta_e)

        if delta_e < 0:
            incumbent = new_solution
            incumbent_cost = new_cost
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e
            if incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
        else:
            stats[op_name]["worse_feasible"] += 1
            if random.random() < 0.8:
                incumbent = new_solution
                incumbent_cost = new_cost
                stats[op_name]["accepted"] += 1
                stats[op_name]["uphill_accepted"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["uphill_accepted_delta_sum"] += delta_e
            else:
                stats[op_name]["uphill_rejected"] += 1

    delta_avg = (sum(deltas) / len(deltas)) if deltas else 1.0
    t0 = -delta_avg / math.log(0.8)
    t0_used_fallback = False
    if t0 <= 0:
        t0 = 1.0
        t0_used_fallback = True
    stats["_meta"] = {
        "warmup_delta_avg": delta_avg,
        "warmup_delta_samples": len(deltas),
        "t0": t0,
        "t0_used_fallback": t0_used_fallback,
    }

    alpha = (final_temperature / t0) ** (1.0 / iterations) if iterations > 0 else 1.0
    temperature = t0

    for it in range(iterations):
        configure_operator_search_progress((warmup_iterations + it) / total_steps)
        new_solution, op_key = apply_weighted_operator(incumbent, operator_weights=normalized_weights)
        op_name = op_key.split("_")[0]
        stats[op_name]["used"] += 1
        if new_solution is incumbent or new_solution == incumbent:
            continue
        if not fast_precheck_solution(new_solution, ctx):
            continue
        feasible, new_cost = cached_evaluate(new_solution)
        if feasible:
            stats[op_name]["feasible"] += 1
            delta_e = new_cost - incumbent_cost

            if delta_e < 0:
                incumbent = new_solution
                incumbent_cost = new_cost
                stats[op_name]["accepted"] += 1
                stats[op_name]["improved"] += 1
                stats[op_name]["delta_sum"] += delta_e
                stats[op_name]["improve_delta_sum"] += -delta_e
                if incumbent_cost < best_cost:
                    best_solution = clone_solution(incumbent)
                    best_cost = incumbent_cost
            else:
                stats[op_name]["worse_feasible"] += 1
                p_accept = math.exp(-delta_e / temperature) if temperature > 0 else 0.0
                stats[op_name]["p_accept_sum"] += p_accept
                stats[op_name]["p_accept_count"] += 1
                if random.random() < p_accept:
                    incumbent = new_solution
                    incumbent_cost = new_cost
                    stats[op_name]["accepted"] += 1
                    stats[op_name]["uphill_accepted"] += 1
                    stats[op_name]["delta_sum"] += delta_e
                    stats[op_name]["uphill_accepted_delta_sum"] += delta_e
                else:
                    stats[op_name]["uphill_rejected"] += 1

        temperature = alpha * temperature

    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    return best_solution, best_cost, stats


def run_statistics(
    initial_solution,
    instance_data=None,
    runs=10,
    warmup_iterations=100,
    iterations=9900,
    final_temperature=0.1,
    cache_limit=200000,
    plot_best_after_all=True,
    operator_weights=None,
    op1_truck_to_drone_bias=None,
    verbose=True,
    return_metrics=False,
    print_solution_pipe=False,
):
    if instance_data is None:
        instance_data = load_instance()
    ctx, calc, checker = build_evaluator(instance_data)
    configure_operator_context(instance_data)
    configure_reinsert_bias(op1_truck_to_drone_bias)

    init_feasible, init_cost = evaluate_solution(initial_solution, calc, checker)
    if not init_feasible:
        raise ValueError("Initial solution is not feasible; cannot compute improvement statistics.")

    run_costs = []
    run_times = []
    global_best_solution = None
    global_best_cost = float("inf")
    shared_eval_cache = OrderedDict()
    aggregate_stats = {
        "op1": {
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
        },
        "op2": {
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
        },
        "op3": {
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
        },
    }
    warmup_delta_avgs = []
    warmup_delta_samples = []
    t0_values = []
    t0_fallback_count = 0

    for run_id in range(runs):
        start = time.perf_counter()
        best_solution, best_cost, op_stats = simulated_annealing(
            clone_solution(initial_solution),
            instance_data=instance_data,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            final_temperature=final_temperature,
            cache_limit=cache_limit,
            operator_weights=operator_weights,
            op1_truck_to_drone_bias=op1_truck_to_drone_bias,
            ctx=ctx,
            calc=calc,
            checker=checker,
            shared_eval_cache=shared_eval_cache,
        )
        elapsed = time.perf_counter() - start
        for op_name in ("op1", "op2", "op3"):
            aggregate_stats[op_name]["used"] += op_stats[op_name]["used"]
            aggregate_stats[op_name]["feasible"] += op_stats[op_name]["feasible"]
            aggregate_stats[op_name]["accepted"] += op_stats[op_name]["accepted"]
            aggregate_stats[op_name]["improved"] += op_stats[op_name]["improved"]
            aggregate_stats[op_name]["uphill_accepted"] += op_stats[op_name]["uphill_accepted"]
            aggregate_stats[op_name]["uphill_rejected"] += op_stats[op_name]["uphill_rejected"]
            aggregate_stats[op_name]["worse_feasible"] += op_stats[op_name]["worse_feasible"]
            aggregate_stats[op_name]["delta_sum"] += op_stats[op_name]["delta_sum"]
            aggregate_stats[op_name]["improve_delta_sum"] += op_stats[op_name]["improve_delta_sum"]
            aggregate_stats[op_name]["uphill_accepted_delta_sum"] += op_stats[op_name][
                "uphill_accepted_delta_sum"
            ]
            aggregate_stats[op_name]["p_accept_sum"] += op_stats[op_name]["p_accept_sum"]
            aggregate_stats[op_name]["p_accept_count"] += op_stats[op_name]["p_accept_count"]
        run_meta = op_stats.get("_meta")
        if run_meta is not None:
            warmup_delta_avgs.append(run_meta.get("warmup_delta_avg", 0.0))
            warmup_delta_samples.append(run_meta.get("warmup_delta_samples", 0))
            t0_values.append(run_meta.get("t0", 0.0))
            if run_meta.get("t0_used_fallback", False):
                t0_fallback_count += 1

        run_costs.append(best_cost)
        run_times.append(elapsed)

        if verbose:
            print(f"Best score in run {run_id + 1}/{runs}: {best_cost}")

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = clone_solution(best_solution)

    avg_obj = sum(run_costs) / len(run_costs)
    best_obj = global_best_cost
    improvement_abs = init_cost - best_obj
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)
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
    }

    if verbose:
        print(f"Average score: {avg_obj}")
        print(f"Best score: {best_obj}")
        print(f"Average runtime: {avg_runtime_per_run:.4f} seconds")
        print(
            f"Improvement over initial solution: {improvement_abs} "
            f"({improvement_pct:.2f}% reduction)"
        )
        if t0_values:
            warmup_delta_avg_mean = sum(warmup_delta_avgs) / len(warmup_delta_avgs)
            warmup_samples_mean = sum(warmup_delta_samples) / len(warmup_delta_samples)
            t0_mean = sum(t0_values) / len(t0_values)
            t0_min = min(t0_values)
            t0_max = max(t0_values)
            print(
                "SA temperature diagnostics: "
                f"warmup_delta_avg_mean={warmup_delta_avg_mean:.4f}, "
                f"warmup_samples_mean={warmup_samples_mean:.1f}, "
                f"t0_mean={t0_mean:.4f}, t0_min={t0_min:.4f}, t0_max={t0_max:.4f}, "
                f"t0_fallback_runs={t0_fallback_count}/{runs}"
            )
        if print_solution_pipe and global_best_solution is not None:
            print("Solution on pipe format:")
            print(format_solution_pipe(global_best_solution))
        print("Operator contribution stats (aggregated):")
        for op_name in ("op1", "op2", "op3"):
            used = aggregate_stats[op_name]["used"]
            feasible_moves = aggregate_stats[op_name]["feasible"]
            accepted = aggregate_stats[op_name]["accepted"]
            improved = aggregate_stats[op_name]["improved"]
            uphill_accepted = aggregate_stats[op_name]["uphill_accepted"]
            uphill_rejected = aggregate_stats[op_name]["uphill_rejected"]
            worse_feasible = aggregate_stats[op_name]["worse_feasible"]
            avg_delta = (
                aggregate_stats[op_name]["delta_sum"] / accepted if accepted > 0 else float("nan")
            )
            feasible_rate = (feasible_moves / used * 100.0) if used > 0 else 0.0
            accept_rate = (accepted / feasible_moves * 100.0) if feasible_moves > 0 else 0.0
            improve_rate = (improved / accepted * 100.0) if accepted > 0 else 0.0
            uphill_rate = (uphill_accepted / accepted * 100.0) if accepted > 0 else 0.0
            uphill_of_worse_rate = (
                uphill_accepted / worse_feasible * 100.0 if worse_feasible > 0 else 0.0
            )
            avg_uphill_accepted_delta = (
                aggregate_stats[op_name]["uphill_accepted_delta_sum"] / uphill_accepted
                if uphill_accepted > 0
                else float("nan")
            )
            mean_p_accept = (
                aggregate_stats[op_name]["p_accept_sum"] / aggregate_stats[op_name]["p_accept_count"]
                if aggregate_stats[op_name]["p_accept_count"] > 0
                else float("nan")
            )
            improve_per_1k_uses = (
                aggregate_stats[op_name]["improve_delta_sum"] * 1000.0 / used
                if used > 0
                else 0.0
            )
            print(
                f"  {op_name}: used={used}, feasible={feasible_moves} ({feasible_rate:.1f}%), "
                f"accepted={accepted} ({accept_rate:.1f}%), improved={improved} ({improve_rate:.1f}%), "
                f"uphill_accepted={uphill_accepted} ({uphill_rate:.1f}%), "
                f"worse_feasible={worse_feasible}, uphill_accept_of_worse={uphill_of_worse_rate:.1f}%, "
                f"uphill_rejected={uphill_rejected}, mean_p_accept={mean_p_accept:.4f}, "
                f"avg_uphill_accepted_delta={avg_uphill_accepted_delta:.4f}, "
                f"avg_accepted_delta={avg_delta:.4f}, improve_per_1000_uses={improve_per_1k_uses:.2f}"
            )

    if plot_best_after_all and global_best_solution is not None:
        plot_solution(
            global_best_solution,
            instance_data,
            title=f"Best Solution After {runs} Runs (score: {global_best_cost})",
        )

    if return_metrics:
        return global_best_solution, global_best_cost, metrics
    return global_best_solution, global_best_cost


def format_solution_pipe(solution):
    parts = to_parts_solution(solution)
    part1 = ",".join(str(x) for x in parts["part1"])
    part2 = ",".join(str(x) for x in parts["part2"])
    part3 = ",".join(str(x) for x in parts["part3"])
    part4 = ",".join(str(x) for x in parts["part4"])
    return f"{part1} | {part2} | {part3} | {part4}"


def main():
    instance_data = load_instance()
    n_customers = instance_data["n_customers"]
    truck_route = [i for i in range(n_customers + 1)] + [0]
    initial_solution = [truck_route, [], []]
    print_solution_pipe_main = False
    configs = configs = [
("OP1 heavy even more balanced", {"op1": 0.643, "op2": 0.319, "op3": 0.038}),
]


    summary = []
    global_best_solution = None
    global_best_cost = float("inf")
    global_best_name = None
    for name, weights in configs:
        print(f"\n=== {name} ===")
        best_solution, best_cost, metrics = run_statistics(
            clone_solution(initial_solution),
            instance_data=instance_data,
            runs=10,
            warmup_iterations=500,
            iterations=9500,
            final_temperature=0.1,
            cache_limit=200000,
            plot_best_after_all=False,
            operator_weights=weights,
            verbose=True,
            return_metrics=True,
            print_solution_pipe=print_solution_pipe_main,
        )
        summary.append((name, metrics["average_score"], best_cost, best_solution))
        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = clone_solution(best_solution)
            global_best_name = name

    print("\n=== Comparison (lower is better) ===")
    for name, avg_score, best_cost, _ in summary:
        print(f"{name}: average={avg_score}, best={best_cost}")

    print(f"Best configuration: {global_best_name}")
    if print_solution_pipe_main:
        print("Best solution (pipe format):")
        print(format_solution_pipe(global_best_solution))
    plot_solution(
        global_best_solution,
        instance_data,
        title=f"Best Solution Across Configs: {global_best_name} (score: {global_best_cost})",
    )


if __name__ == "__main__":
    main()
