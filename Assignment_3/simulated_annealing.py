from read_file import read_instance
from op import operator, set_operator_context
from pathlib import Path
import sys
import importlib.util
import copy
import random
import math
import time

GENERAL_ALGORITHM_PATH = Path(__file__).resolve().parents[1] / "General_algorithm"
if str(GENERAL_ALGORITHM_PATH) not in sys.path:
    sys.path.append(str(GENERAL_ALGORITHM_PATH))

try:
    from CalCulateTotalArrivalTime import CalCulateTotalArrivalTime
    from FeasibiltyCheck import SolutionFeasibility
except ModuleNotFoundError:
    calc_path = GENERAL_ALGORITHM_PATH / "CalCulateTotalArrivalTime.py"
    feas_path = GENERAL_ALGORITHM_PATH / "FeasibiltyCheck.py"

    calc_spec = importlib.util.spec_from_file_location("CalCulateTotalArrivalTime", calc_path)
    calc_mod = importlib.util.module_from_spec(calc_spec)
    calc_spec.loader.exec_module(calc_mod)
    CalCulateTotalArrivalTime = calc_mod.CalCulateTotalArrivalTime

    feas_spec = importlib.util.spec_from_file_location("FeasibiltyCheck", feas_path)
    feas_mod = importlib.util.module_from_spec(feas_spec)
    feas_spec.loader.exec_module(feas_mod)
    SolutionFeasibility = feas_mod.SolutionFeasibility

absolute_path = "/Users/niklasmarton/Library/CloudStorage/OneDrive-Personlig/ITØK/Metaheuristics/TrucksAndDrones/Test_files/"
file_name = "R_10.txt"

instance = read_instance(f"{absolute_path}{file_name}")

T = instance["truck_times"]
D = instance["drone_times"]
flight_limit = instance["flight_limit"]
n_customers = instance["n_customers"]
depot = instance.get("depot_index", 0)

set_operator_context(T, D, flight_limit, depot)


def to_parts_solution(solution):
    """
    Convert tuple-based solution format:
    [truck, drone1[(node, launch_idx, land_idx)], drone2[(node, launch_idx, land_idx)]]
    to the dictionary format expected by feasibility and objective functions.
    """
    truck, drone1, drone2 = solution

    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]

    # Feasibility/evaluation modules use 1-based launch/landing cells.
    launch_indices_1 = [launch_idx + 1 for _, launch_idx, _ in drone1]
    landing_indices_1 = [land_idx + 1 for _, _, land_idx in drone1]
    launch_indices_2 = [launch_idx + 1 for _, launch_idx, _ in drone2]
    landing_indices_2 = [land_idx + 1 for _, _, land_idx in drone2]

    drone_serving = drone_serving_1 + [-1] + drone_serving_2
    drone_total_launches = launch_indices_1 + [-1] + launch_indices_2
    drone_total_landings = landing_indices_1 + [-1] + landing_indices_2

    return {
        "part1": truck,
        "part2": drone_serving,
        "part3": drone_total_launches,
        "part4": drone_total_landings,
    }


def evaluate_solution(solution, calc, checker):
    parts_solution = to_parts_solution(solution)
    # Structural/route feasibility check.
    if not checker.is_solution_feasible(parts_solution):
        return False, float("inf")

    # Timing-based feasibility (captures range violations with synchronization waiting).
    total_time, _, _, calc_feasible = calc.calculate_total_waiting_time(parts_solution)
    if not calc_feasible:
        return False, float("inf")

    return True, total_time


def simulated_annealing(initial_solution, warmup_iterations=100, iterations=9900, final_temperature=0.1):
    """
    Simulated annealing per assignment pseudocode.

    Warm-up phase (100 iters):
      - collect DeltaE samples (feasible moves only)
      - accept improving moves
      - accept non-improving moves with probability 0.8

    Main SA phase (9900 iters):
      - accept improving moves
      - accept worsening moves with probability exp(-DeltaE/T)
      - cool with T <- alpha * T
    """
    calc = CalCulateTotalArrivalTime()
    calc.truck_times = T
    calc.drone_times = D
    calc.flight_range = flight_limit
    calc.depot_index = depot
    # Ensure operator uses the same active instance as evaluator/checker.
    set_operator_context(T, D, flight_limit, depot)

    checker = SolutionFeasibility(
        n_nodes=n_customers + 1,
        n_drones=2,
        depot_index=depot,
        drone_times=D,
        flight_range=flight_limit,
    )

    incumbent = copy.deepcopy(initial_solution)
    incumbent_feasible, incumbent_cost = evaluate_solution(incumbent, calc, checker)
    if not incumbent_feasible:
        raise ValueError("Initial solution is not feasible.")

    best_solution = copy.deepcopy(incumbent)
    best_cost = incumbent_cost

    deltas = []

    # Warm-up phase: w = 1..100
    for _ in range(warmup_iterations):
        new_solution = operator(incumbent)
        feasible, new_cost = evaluate_solution(new_solution, calc, checker)
        if not feasible:
            continue

        delta_e = new_cost - incumbent_cost
        deltas.append(delta_e)

        if delta_e < 0:
            incumbent = new_solution
            incumbent_cost = new_cost
            if incumbent_cost < best_cost:
                best_solution = copy.deepcopy(incumbent)
                best_cost = incumbent_cost
        else:
            if random.random() < 0.8:
                incumbent = new_solution
                incumbent_cost = new_cost

    # Compute T0 and alpha from pseudocode.
    if deltas:
        delta_avg = sum(deltas) / len(deltas)
    else:
        delta_avg = 1.0

    # T0 = -DeltaAvg / ln(0.8)
    t0 = -delta_avg / math.log(0.8)
    # Guard for degenerate/non-positive temperature from unusual delta samples.
    if t0 <= 0:
        t0 = 1.0

    # alpha = (Tf / T0)^(1/iterations)
    if iterations > 0:
        alpha = (final_temperature / t0) ** (1.0 / iterations)
    else:
        alpha = 1.0

    temperature = t0

    # Main SA phase: iteration = 1..9900
    for _ in range(iterations):
        new_solution = operator(incumbent)
        feasible, new_cost = evaluate_solution(new_solution, calc, checker)
        if feasible:
            delta_e = new_cost - incumbent_cost

            if delta_e < 0:
                incumbent = new_solution
                incumbent_cost = new_cost
                if incumbent_cost < best_cost:
                    best_solution = copy.deepcopy(incumbent)
                    best_cost = incumbent_cost
            else:
                # p = exp(-DeltaE / T)
                if temperature > 0:
                    p_accept = math.exp(-delta_e / temperature)
                else:
                    p_accept = 0.0

                if random.random() < p_accept:
                    incumbent = new_solution
                    incumbent_cost = new_cost

        temperature = alpha * temperature

    # Safety: never return an infeasible best.
    best_parts = to_parts_solution(best_solution)
    assert checker.is_solution_feasible(best_parts)
    assert calc.calculate_total_waiting_time(best_parts)[3]

    return best_solution, best_cost


def run_statistics(initial_solution, runs=10, warmup_iterations=100, iterations=9900, final_temperature=0.1):
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

    init_feasible, init_cost = evaluate_solution(initial_solution, calc, checker)
    if not init_feasible:
        raise ValueError("Initial solution is not feasible; cannot compute improvement statistics.")

    run_costs = []
    run_times = []
    global_best_solution = None
    global_best_cost = float("inf")

    for run_id in range(runs):
        random.seed(run_id)
        start = time.perf_counter()
        best_solution, best_cost = simulated_annealing(
            copy.deepcopy(initial_solution),
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            final_temperature=final_temperature,
        )
        elapsed = time.perf_counter() - start

        run_costs.append(best_cost)
        run_times.append(elapsed)

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_solution = best_solution

    avg_obj = sum(run_costs) / len(run_costs)
    best_obj = global_best_cost
    improvement_abs = init_cost - best_obj
    improvement_pct = (improvement_abs / init_cost * 100.0) if init_cost > 0 else 0.0
    avg_runtime_per_run = sum(run_times) / len(run_times)

    print(f"Initial objective value: {init_cost}")
    print(f"Average objective value (over {runs} runs): {avg_obj}")
    print(f"Best objective: {best_obj}")
    print(
        f"Improvement (initial -> best): {improvement_abs} "
        f"({improvement_pct:.2f}% reduction)"
    )
    print(
        f"Average runtime per iteration (1 iteration = SA run with {warmup_iterations + iterations} loops): "
        f"{avg_runtime_per_run:.4f} seconds"
    )

    return global_best_solution, global_best_cost


def format_solution_pipe(solution):
    # Display format uses internal (0-based) launch/landing indices.
    truck, drone1, drone2 = solution
    drone_serving_1 = [node for node, _, _ in drone1]
    drone_serving_2 = [node for node, _, _ in drone2]
    launch_indices_1 = [launch_idx for _, launch_idx, _ in drone1]
    launch_indices_2 = [launch_idx for _, launch_idx, _ in drone2]
    landing_indices_1 = [land_idx for _, _, land_idx in drone1]
    landing_indices_2 = [land_idx for _, _, land_idx in drone2]

    part1 = ",".join(str(x) for x in truck)
    part2 = ",".join(str(x) for x in (drone_serving_1 + [-1] + drone_serving_2))
    part3 = ",".join(str(x) for x in (launch_indices_1 + [-1] + launch_indices_2))
    part4 = ",".join(str(x) for x in (landing_indices_1 + [-1] + landing_indices_2))
    return f"{part1} | {part2} | {part3} | {part4}"


def format_solution_pipe_checker(solution):
    """
    Checker-compatible display format:
    - drone served nodes unchanged
    - launch/landing cells shown as 1-based indices
    """
    parts = to_parts_solution(solution)
    part1 = ",".join(str(x) for x in parts["part1"])
    part2 = ",".join(str(x) for x in parts["part2"])
    part3 = ",".join(str(x) for x in parts["part3"])
    part4 = ",".join(str(x) for x in parts["part4"])
    return f"{part1} | {part2} | {part3} | {part4}"


if __name__ == "__main__":
    truck_route = [i for i in range(n_customers + 1)] + [0]
    drone1 = []
    drone2 = []
    initial_solution = [truck_route, drone1, drone2]

    print(f"Old solution was: {initial_solution}")

    best_solution, best_cost = run_statistics(
        initial_solution,
        runs=10,
        warmup_iterations=100,
        iterations=9900,
        final_temperature=0.1,
    )
    print(f"Best solution after all runs: {best_solution}")
    print(f"Best solution (pipe format, internal 0-based): {format_solution_pipe(best_solution)}")
    print(f"Best solution (pipe format, checker 1-based): {format_solution_pipe_checker(best_solution)}")
