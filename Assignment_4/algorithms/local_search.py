from collections import OrderedDict
from pathlib import Path
import sys
import time

# Ensure this file can import the weighted SA module when run directly.
ALG_DIR = Path(__file__).resolve().parent
if str(ALG_DIR) not in sys.path:
    sys.path.append(str(ALG_DIR))

import simulated_annealing_3op_weighted as sa_weighted


def build_index_order_truck_only_initial_solution(instance_data):
    n_customers = instance_data["n_customers"]
    truck_route = [i for i in range(n_customers + 1)] + [0]
    return [truck_route, [], []]


def solution_key(solution):
    truck, drone1, drone2 = solution
    return (tuple(truck), tuple(drone1), tuple(drone2))


def local_search(
    initial_solution,
    instance_data=None,
    iterations=10000,
    stall_limit=2500,
    cache_limit=200000,
    operator_weights={"op1": 0.2, "op2": 0.5, "op3": 0.3},
    ctx=None,
    calc=None,
    checker=None,
    shared_eval_cache=None,
):
    """
    Weighted first-improvement local search (hill climbing):
    - Select one operator per iteration using the same weighted chooser as SA.
    - Accept only strictly improving feasible moves (delta < 0).
    - Stop when iteration budget is exhausted or no improvement is found for
      `stall_limit` consecutive iterations.
    """
    if instance_data is None:
        instance_data = sa_weighted.load_instance()
    if ctx is None or calc is None or checker is None:
        ctx, calc, checker = sa_weighted.build_evaluator(instance_data)
        sa_weighted.configure_operator_context(instance_data)

    normalized_weights = sa_weighted._normalize_operator_weights(operator_weights)

    eval_cache = shared_eval_cache if shared_eval_cache is not None else OrderedDict()
    stats = {
        "op1": {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
        "op2": {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
        "op3": {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
    }

    def cached_evaluate(sol):
        key = solution_key(sol)
        cached = eval_cache.get(key)
        if cached is not None:
            eval_cache.move_to_end(key)
            return cached

        result = sa_weighted.evaluate_solution(sol, calc, checker)
        eval_cache[key] = result
        if cache_limit is not None and cache_limit > 0 and len(eval_cache) > cache_limit:
            eval_cache.popitem(last=False)
        return result

    incumbent = sa_weighted.clone_solution(initial_solution)
    incumbent_feasible, incumbent_cost = cached_evaluate(incumbent)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = sa_weighted.clone_solution(incumbent)
    best_cost = incumbent_cost
    no_improve_steps = 0

    for it in range(iterations):
        sa_weighted.configure_operator_search_progress(it / max(1, iterations))
        new_solution, op_key = sa_weighted.apply_weighted_operator(
            incumbent,
            operator_weights=normalized_weights,
        )
        op_name = op_key.split("_")[0]
        stats[op_name]["used"] += 1

        if new_solution is incumbent or new_solution == incumbent:
            no_improve_steps += 1
            if stall_limit is not None and no_improve_steps >= stall_limit:
                break
            continue

        if not sa_weighted.fast_precheck_solution(new_solution, ctx):
            no_improve_steps += 1
            if stall_limit is not None and no_improve_steps >= stall_limit:
                break
            continue

        feasible, new_cost = cached_evaluate(new_solution)
        if not feasible:
            no_improve_steps += 1
            if stall_limit is not None and no_improve_steps >= stall_limit:
                break
            continue

        stats[op_name]["feasible"] += 1
        delta_e = new_cost - incumbent_cost

        # Standard hill-climbing acceptance rule: only strictly better moves.
        if delta_e < 0:
            incumbent = new_solution
            incumbent_cost = new_cost
            stats[op_name]["accepted"] += 1
            stats[op_name]["improved"] += 1
            stats[op_name]["delta_sum"] += delta_e
            stats[op_name]["improve_delta_sum"] += -delta_e
            no_improve_steps = 0

            if incumbent_cost < best_cost:
                best_solution = sa_weighted.clone_solution(incumbent)
                best_cost = incumbent_cost
        else:
            no_improve_steps += 1

        if stall_limit is not None and no_improve_steps >= stall_limit:
            break

    best_parts = sa_weighted.to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    return best_solution, best_cost, stats


def run_statistics(
    initial_solution,
    instance_data=None,
    runs=10,
    iterations=10000,
    stall_limit=2500,
    cache_limit=200000,
    plot_best_after_all=True,
    operator_weights=None,
    verbose=True,
    return_metrics=False,
    print_solution_pipe=False,
):
    if instance_data is None:
        instance_data = sa_weighted.load_instance()

    ctx, calc, checker = sa_weighted.build_evaluator(instance_data)
    sa_weighted.configure_operator_context(instance_data)

    init_feasible, init_cost = sa_weighted.evaluate_solution(initial_solution, calc, checker)
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
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
        "op2": {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
        "op3": {
            "used": 0,
            "feasible": 0,
            "accepted": 0,
            "improved": 0,
            "delta_sum": 0.0,
            "improve_delta_sum": 0.0,
        },
    }

    for run_id in range(runs):
        start = time.perf_counter()
        best_solution, best_cost, op_stats = local_search(
            sa_weighted.clone_solution(initial_solution),
            instance_data=instance_data,
            iterations=iterations,
            stall_limit=stall_limit,
            cache_limit=cache_limit,
            operator_weights=operator_weights,
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
            aggregate_stats[op_name]["delta_sum"] += op_stats[op_name]["delta_sum"]
            aggregate_stats[op_name]["improve_delta_sum"] += op_stats[op_name]["improve_delta_sum"]

        run_costs.append(best_cost)
        run_times.append(elapsed)

        if verbose:
            print(f"Best score in run {run_id + 1}/{runs}: {best_cost}")

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = sa_weighted.clone_solution(best_solution)

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
    }

    if verbose:
        print(f"Average score: {avg_obj}")
        print(f"Best score: {best_obj}")
        print(f"Average runtime: {avg_runtime_per_run:.4f} seconds")
        print(
            f"Improvement over initial solution: {improvement_abs} "
            f"({improvement_pct:.2f}% reduction)"
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
            avg_delta = (
                aggregate_stats[op_name]["delta_sum"] / accepted if accepted > 0 else float("nan")
            )
            feasible_rate = (feasible_moves / used * 100.0) if used > 0 else 0.0
            accept_rate = (accepted / feasible_moves * 100.0) if feasible_moves > 0 else 0.0
            improve_rate = (improved / accepted * 100.0) if accepted > 0 else 0.0
            improve_per_1k_uses = (
                aggregate_stats[op_name]["improve_delta_sum"] * 1000.0 / used if used > 0 else 0.0
            )
            print(
                f"  {op_name}: used={used}, feasible={feasible_moves} ({feasible_rate:.1f}%), "
                f"accepted={accepted} ({accept_rate:.1f}%), improved={improved} ({improve_rate:.1f}%), "
                f"avg_accepted_delta={avg_delta:.4f}, improve_per_1000_uses={improve_per_1k_uses:.2f}"
            )

    if plot_best_after_all and global_best_solution is not None:
        sa_weighted.plot_solution(
            global_best_solution,
            instance_data,
            title=f"Best Local Search Solution After {runs} Runs (score: {global_best_cost})",
        )

    if return_metrics:
        return global_best_solution, global_best_cost, metrics
    return global_best_solution, global_best_cost


def format_solution_pipe(solution):
    parts = sa_weighted.to_parts_solution(solution)
    part1 = ",".join(str(x) for x in parts["part1"])
    part2 = ",".join(str(x) for x in parts["part2"])
    part3 = ",".join(str(x) for x in parts["part3"])
    part4 = ",".join(str(x) for x in parts["part4"])
    return f"{part1} | {part2} | {part3} | {part4}"


def main():
    instance_data = sa_weighted.load_instance()
    initial_solution = build_index_order_truck_only_initial_solution(instance_data)
    print_solution_pipe_main = False

    configs = [
        ("OP1 heavy even more balanced", {"op1": 0.2, "op2": 0.75, "op3": 0.05}),       
    ]

    summary = []
    global_best_solution = None
    global_best_cost = float("inf")
    global_best_name = None

    for name, weights in configs:
        print(f"\n=== {name} ===")
        best_solution, best_cost, metrics = run_statistics(
            sa_weighted.clone_solution(initial_solution),
            instance_data=instance_data,
            runs=10,
            iterations=10000,
            stall_limit=2500,
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
            global_best_solution = sa_weighted.clone_solution(best_solution)
            global_best_name = name

    print("\n=== Comparison (lower is better) ===")
    for name, avg_score, best_cost, _ in summary:
        print(f"{name}: average={avg_score}, best={best_cost}")

    if global_best_solution is not None:
        print(f"Best configuration: {global_best_name}")
        if print_solution_pipe_main:
            print("Best solution (pipe format):")
            print(format_solution_pipe(global_best_solution))


if __name__ == "__main__":
    main()
