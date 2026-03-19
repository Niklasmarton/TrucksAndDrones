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
import op2_destroy_repair as op2
import op3_or_opt as op3
from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
from FeasibiltyCheck import SolutionFeasibility

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


def reset_operator_state():
    if hasattr(op1, "reset_operator_state"):
        op1.reset_operator_state()


def configure_reinsert_bias(op1_truck_to_drone_bias=None):
    if op1_truck_to_drone_bias is None:
        return
    if hasattr(op1, "set_truck_to_drone_bias"):
        op1.set_truck_to_drone_bias(op1_truck_to_drone_bias)


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


def format_solution_pipe(solution):
    parts = to_parts_solution(solution)
    part1 = ",".join(str(x) for x in parts["part1"])
    part2 = ",".join(str(x) for x in parts["part2"])
    part3 = ",".join(str(x) for x in parts["part3"])
    part4 = ",".join(str(x) for x in parts["part4"])
    return f"{part1} | {part2} | {part3} | {part4}"


def apply_operator(solution, op_name):
    if op_name == "op1":
        return op1.operator(solution)
    if op_name == "op2":
        return op2.operator(solution)
    return op3.truck_2opt(solution)


def roulette_pick(weights):
    r = random.random()
    acc = 0.0
    for op_name in ("op1", "op2", "op3"):
        acc += weights[op_name]
        if r <= acc:
            return op_name
    return "op3"


def _normalize_weight_dict(weights, min_weight=1e-9):
    fixed = {k: max(min_weight, float(v)) for k, v in weights.items()}
    s = sum(fixed.values())
    return {k: v / s for k, v in fixed.items()}


def _escape_perturbation(
    incumbent,
    incumbent_cost,
    best_cost,
    ctx,
    cached_evaluate,
    weights,
    escape_steps=10,
):
    improved_best = False
    total_applied = 0

    current = incumbent
    current_cost = incumbent_cost

    for _ in range(escape_steps):
        op_name = roulette_pick(weights)
        candidate = apply_operator(current, op_name)
        if candidate is current or candidate == current:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue
        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue

        current = candidate
        current_cost = cand_cost
        total_applied += 1

        if current_cost < best_cost:
            improved_best = True
            break

    return current, current_cost, improved_best, total_applied


def alns(
    initial_solution,
    instance_data=None,
    warmup_iterations=500,
    iterations=9500,
    final_temperature=0.1,
    cache_limit=200000,
    op1_truck_to_drone_bias=None,
    reaction_factor=0.2,
    segment_length=100,
    escape_stall_limit=800,
    escape_steps=10,
    ctx=None,
    calc=None,
    checker=None,
    shared_eval_cache=None,
):
    if instance_data is None:
        instance_data = load_instance()
    if ctx is None or calc is None or checker is None:
        ctx, calc, checker = build_evaluator(instance_data)
        configure_operator_context(instance_data)

    configure_reinsert_bias(op1_truck_to_drone_bias)
    reset_operator_state()

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

    incumbent = clone_solution(initial_solution)
    incumbent_feasible, incumbent_cost = cached_evaluate(incumbent)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = clone_solution(incumbent)
    best_cost = incumbent_cost

    total_steps = max(1, warmup_iterations + iterations)

    deltas = []
    for w in range(warmup_iterations):
        configure_operator_search_progress(w / total_steps)
        op_name = random.choice(("op1", "op2", "op3"))
        stats[op_name]["used"] += 1

        candidate = apply_operator(incumbent, op_name)
        if candidate is incumbent or candidate == incumbent:
            continue
        if not fast_precheck_solution(candidate, ctx):
            continue

        feasible, cand_cost = cached_evaluate(candidate)
        if not feasible:
            continue
        stats[op_name]["feasible"] += 1

        delta_e = cand_cost - incumbent_cost
        if delta_e >= 0:
            deltas.append(delta_e)

        if delta_e < 0:
            incumbent = candidate
            incumbent_cost = cand_cost
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
                incumbent = candidate
                incumbent_cost = cand_cost
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

    alpha = (final_temperature / t0) ** (1.0 / iterations) if iterations > 0 else 1.0
    temperature = t0

    weights = {"op1": 1.0 / 3.0, "op2": 1.0 / 3.0, "op3": 1.0 / 3.0}
    segment_scores = {"op1": 0.0, "op2": 0.0, "op3": 0.0}
    segment_uses = {"op1": 0, "op2": 0, "op3": 0}
    segment_feasible_uses = {"op1": 0, "op2": 0, "op3": 0}

    sigma_global_best = 4.0
    sigma_incumbent_improve = 2.0
    sigma_feasible = 1.0

    escape_calls = 0
    escape_steps_applied = 0
    no_best_improve_steps = 0

    for it in range(iterations):
        configure_operator_search_progress((warmup_iterations + it) / total_steps)
        op_name = roulette_pick(weights)
        stats[op_name]["used"] += 1
        segment_uses[op_name] += 1

        candidate = apply_operator(incumbent, op_name)
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
        else:
            stats[op_name]["worse_feasible"] += 1
            p_accept = math.exp(-delta_e / temperature) if temperature > 0 else 0.0
            stats[op_name]["p_accept_sum"] += p_accept
            stats[op_name]["p_accept_count"] += 1
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

        if improved_best:
            segment_scores[op_name] += sigma_global_best
            no_best_improve_steps = 0
        elif delta_e < 0:
            segment_scores[op_name] += sigma_incumbent_improve
            no_best_improve_steps += 1
        else:
            segment_scores[op_name] += sigma_feasible
            no_best_improve_steps += 1

        if (it + 1) % segment_length == 0:
            updated = {}
            for k in ("op1", "op2", "op3"):
                theta = segment_feasible_uses[k]
                if theta > 0:
                    updated[k] = weights[k] * (1.0 - reaction_factor) + reaction_factor * (
                        segment_scores[k] / theta
                    )
                else:
                    updated[k] = weights[k] * (1.0 - reaction_factor)
            weights = _normalize_weight_dict(updated)
            segment_scores = {"op1": 0.0, "op2": 0.0, "op3": 0.0}
            segment_uses = {"op1": 0, "op2": 0, "op3": 0}
            segment_feasible_uses = {"op1": 0, "op2": 0, "op3": 0}

        if no_best_improve_steps >= escape_stall_limit:
            escape_calls += 1
            incumbent, incumbent_cost, esc_improved_best, steps_applied = _escape_perturbation(
                incumbent,
                incumbent_cost,
                best_cost,
                ctx,
                cached_evaluate,
                weights,
                escape_steps=escape_steps,
            )
            escape_steps_applied += steps_applied
            if esc_improved_best and incumbent_cost < best_cost:
                best_solution = clone_solution(incumbent)
                best_cost = incumbent_cost
                no_best_improve_steps = 0
            else:
                no_best_improve_steps = max(0, escape_stall_limit // 4)

        temperature = alpha * temperature

    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    stats["_meta"] = {
        "warmup_delta_avg": delta_avg,
        "warmup_delta_samples": len(deltas),
        "t0": t0,
        "t0_used_fallback": t0_used_fallback,
        "final_weights": dict(weights),
        "escape_calls": escape_calls,
        "escape_steps_applied": escape_steps_applied,
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
    plot_best_after_all=False,
    op1_truck_to_drone_bias=None,
    reaction_factor=0.2,
    segment_length=100,
    escape_stall_limit=800,
    escape_steps=10,
    verbose=True,
    return_metrics=False,
    print_solution_pipe=True,
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
        for op_name in ("op1", "op2", "op3")
    }

    warmup_delta_avgs = []
    warmup_delta_samples = []
    t0_values = []
    t0_fallback_count = 0
    final_weights_acc = {"op1": 0.0, "op2": 0.0, "op3": 0.0}
    total_escape_calls = 0
    total_escape_steps = 0

    for run_id in range(runs):
        start = time.perf_counter()
        best_solution, best_cost, op_stats = alns(
            clone_solution(initial_solution),
            instance_data=instance_data,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            final_temperature=final_temperature,
            cache_limit=cache_limit,
            op1_truck_to_drone_bias=op1_truck_to_drone_bias,
            reaction_factor=reaction_factor,
            segment_length=segment_length,
            escape_stall_limit=escape_stall_limit,
            escape_steps=escape_steps,
            ctx=ctx,
            calc=calc,
            checker=checker,
            shared_eval_cache=shared_eval_cache,
        )
        elapsed = time.perf_counter() - start

        for op_name in ("op1", "op2", "op3"):
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
            for k in ("op1", "op2", "op3"):
                final_weights_acc[k] += float(fw.get(k, 0.0))
            total_escape_calls += int(run_meta.get("escape_calls", 0))
            total_escape_steps += int(run_meta.get("escape_steps_applied", 0))

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

    avg_final_weights = {k: (final_weights_acc[k] / runs if runs > 0 else 0.0) for k in final_weights_acc}

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
        "escape_calls": total_escape_calls,
        "escape_steps": total_escape_steps,
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
            print(
                "SA temperature diagnostics: "
                f"warmup_delta_avg_mean={sum(warmup_delta_avgs)/len(warmup_delta_avgs):.4f}, "
                f"warmup_samples_mean={sum(warmup_delta_samples)/len(warmup_delta_samples):.1f}, "
                f"t0_mean={sum(t0_values)/len(t0_values):.4f}, "
                f"t0_min={min(t0_values):.4f}, t0_max={max(t0_values):.4f}, "
                f"t0_fallback_runs={t0_fallback_count}/{runs}"
            )
        print(
            "ALNS adaptation diagnostics: "
            f"avg_final_weights=(op1={avg_final_weights['op1']:.3f}, "
            f"op2={avg_final_weights['op2']:.3f}, op3={avg_final_weights['op3']:.3f}), "
            f"escape_calls={total_escape_calls}, escape_steps={total_escape_steps}"
        )
        if print_solution_pipe and global_best_solution is not None:
            print("Solution on pipe format:")
            print(format_solution_pipe(global_best_solution))

        print("Operator contribution stats (aggregated):")
        for op_name in ("op1", "op2", "op3"):
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
            mean_p_accept = s["p_accept_sum"] / s["p_accept_count"] if s["p_accept_count"] > 0 else float("nan")
            improve_per_1k_uses = (s["improve_delta_sum"] * 1000.0 / used) if used > 0 else 0.0

            print(
                f"  {op_name}: used={used}, feasible={feasible_moves} ({feasible_rate:.1f}%), "
                f"accepted={accepted} ({accept_rate:.1f}%), improved={improved} ({improve_rate:.1f}%), "
                f"uphill_accepted={uphill_accepted} ({uphill_rate:.1f}%), "
                f"worse_feasible={worse_feasible}, uphill_accept_of_worse={uphill_of_worse_rate:.1f}%, "
                f"uphill_rejected={uphill_rejected}, mean_p_accept={mean_p_accept:.4f}, "
                f"avg_uphill_accepted_delta={avg_uphill_accepted_delta:.4f}, "
                f"avg_accepted_delta={avg_delta:.4f}, improve_per_1000_uses={improve_per_1k_uses:.2f}"
            )

    if return_metrics:
        return global_best_solution, global_best_cost, metrics
    return global_best_solution, global_best_cost


def main():
    instance_data = load_instance()
    n_customers = instance_data["n_customers"]
    truck_route = [i for i in range(n_customers + 1)] + [0]
    initial_solution = [truck_route, [], []]

    run_statistics(
        initial_solution,
        instance_data=instance_data,
        runs=10,
        warmup_iterations=500,
        iterations=9500,
        final_temperature=0.1,
        cache_limit=200000,
        op1_truck_to_drone_bias=None,
        reaction_factor=0.2,
        segment_length=100,
        escape_stall_limit=800,
        escape_steps=10,
        verbose=True,
        return_metrics=False,
        print_solution_pipe=True,
    )


if __name__ == "__main__":
    main()
